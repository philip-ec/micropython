#!/usr/bin/env python3
"""
Generate modweights.c from a JSON weight spec.

Input JSON format:
{
  "layer_name": {
    "weights": [[row0_col0, row0_col1, ...], [row1_col0, ...], ...],
    "rows": <int>,
    "cols": <int>
  },
  ...
}

Values must be in [-128, 127] (int8 range).

Usage:
    python3 weights_gen.py weights.json
    # generates modweights.c in the same directory
    # then: make && eff-flash build/firmware.hex sram -p /dev/ttyACM0
"""
import json
import sys
import os

def clamp_int8(v):
    return max(-128, min(127, int(v)))

def generate(spec_path):
    with open(spec_path) as f:
        spec = json.load(f)

    lines = []
    lines.append('#include <stdint.h>')
    lines.append('#include "py/obj.h"')
    lines.append('#include "py/runtime.h"')
    lines.append('')
    lines.append('// Fabric kernel — defined in fabric_kernel.c')
    lines.append('void matvec_int8_bias_rq(const int8_t *W, const int8_t *x, const int32_t *bias,')
    lines.append('                          int8_t *out, int32_t rows, int32_t cols,')
    lines.append('                          int32_t scale, int32_t shift, int32_t zero_point);')
    lines.append('')

    layer_names = list(spec.keys())

    # emit frozen weight and bias arrays
    for name, layer in spec.items():
        weights = layer['weights']
        rows = layer['rows']
        cols = layer['cols']
        flat = [clamp_int8(v) for row in weights for v in row]
        assert len(flat) == rows * cols, f"{name}: expected {rows*cols} values, got {len(flat)}"
        lines.append(f'static const int8_t weights_{name}[{rows * cols}] = {{')
        for i in range(0, len(flat), 16):
            chunk = flat[i:i+16]
            lines.append('    ' + ', '.join(str(v) for v in chunk) + ',')
        lines.append('};')
        lines.append('')
        if 'bias' in layer:
            bias = [int(v) for v in layer['bias']]
            assert len(bias) == rows, f"{name}: expected {rows} bias values, got {len(bias)}"
            lines.append(f'static const int32_t bias_{name}[{rows}] = {{')
            for i in range(0, len(bias), 8):
                chunk = bias[i:i+8]
                lines.append('    ' + ', '.join(str(v) for v in chunk) + ',')
            lines.append('};')
            lines.append('')

    # emit one Python-callable glue function per layer
    # signature: weights.NAME(x, scale, shift, zero_point) -> list of int8
    for name, layer in spec.items():
        rows = layer['rows']
        cols = layer['cols']
        has_bias = 'bias' in layer
        lines.append(f'// weights.{name}(x, scale, shift, zero_point)')
        lines.append(f'// x: list of {cols} int8 values; returns list of {rows} int8 values')
        lines.append(f'static mp_obj_t weights_fn_{name}(size_t n_args, const mp_obj_t *args) {{')
        lines.append(f'    (void)n_args;')
        lines.append(f'    size_t x_len; mp_obj_t *x_items;')
        lines.append(f'    mp_obj_get_array(args[0], &x_len, &x_items);')
        lines.append(f'    if ((int32_t)x_len != {cols})')
        lines.append(f'        mp_raise_ValueError(MP_ERROR_TEXT("input must have {cols} elements"));')
        lines.append(f'    int32_t scale = mp_obj_get_int(args[1]);')
        lines.append(f'    int32_t shift = mp_obj_get_int(args[2]);')
        lines.append(f'    int32_t zp    = mp_obj_get_int(args[3]);')
        lines.append(f'    int8_t x_buf[{cols}], out_buf[{rows}];')
        lines.append(f'    for (int32_t i = 0; i < {cols}; i++)')
        lines.append(f'        x_buf[i] = (int8_t)mp_obj_get_int(x_items[i]);')
        if has_bias:
            lines.append(f'    matvec_int8_bias_rq(weights_{name}, x_buf, bias_{name}, out_buf,')
            lines.append(f'                        {rows}, {cols}, scale, shift, zp);')
        else:
            # no bias: use a zero-filled bias (rare case)
            lines.append(f'    static const int32_t _zero_{name}[{rows}] = {{0}};')
            lines.append(f'    matvec_int8_bias_rq(weights_{name}, x_buf, _zero_{name}, out_buf,')
            lines.append(f'                        {rows}, {cols}, scale, shift, zp);')
        lines.append(f'    mp_obj_t result = mp_obj_new_list({rows}, NULL);')
        lines.append(f'    mp_obj_list_t *lst = MP_OBJ_TO_PTR(result);')
        lines.append(f'    for (int32_t i = 0; i < {rows}; i++)')
        lines.append(f'        lst->items[i] = mp_obj_new_int((int32_t)out_buf[i]);')
        lines.append(f'    return result;')
        lines.append(f'}}')
        lines.append(f'static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(weights_fn_{name}_obj, 4, 4, weights_fn_{name});')
        lines.append('')

    # globals table
    lines.append('static const mp_rom_map_elem_t weights_module_globals_table[] = {')
    lines.append('    { MP_ROM_QSTR(MP_QSTR___name__), MP_ROM_QSTR(MP_QSTR_weights) },')
    for name in layer_names:
        lines.append(f'    {{ MP_ROM_QSTR(MP_QSTR_{name}), MP_ROM_PTR(&weights_fn_{name}_obj) }},')
    lines.append('};')
    lines.append('static MP_DEFINE_CONST_DICT(weights_module_globals, weights_module_globals_table);')
    lines.append('')
    lines.append('const mp_obj_module_t weights_module = {')
    lines.append('    .base = { &mp_type_module },')
    lines.append('    .globals = (mp_obj_dict_t *)&weights_module_globals,')
    lines.append('};')
    lines.append('MP_REGISTER_MODULE(MP_QSTR_weights, weights_module);')

    out_path = os.path.join(os.path.dirname(spec_path), 'modweights.c')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"Generated {out_path}")
    layer_summary = ', '.join('{} ({}x{})'.format(n, spec[n]['rows'], spec[n]['cols']) for n in layer_names)
    print(f"Layers: {layer_summary}")
    print("Next: make && eff-flash build/firmware.hex sram -p /dev/ttyACM0")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    generate(sys.argv[1])
