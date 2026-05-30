#!/usr/bin/env python3
"""
Generate modtestdata.c from test_digits.json.
Freezes 10 MNIST test digits (one per class) into firmware as int8 arrays.

Usage:
    python3 testdata_gen.py test_digits.json
    # then rebuild and reflash
"""
import json, sys, os

def generate(spec_path):
    with open(spec_path) as f:
        spec = json.load(f)

    lines = []
    lines.append('#include <stdint.h>')
    lines.append('#include "py/obj.h"')
    lines.append('#include "py/runtime.h"')
    lines.append('')

    n = len(spec)
    size = 784  # MNIST image size

    # emit one frozen array per digit
    for key, entry in sorted(spec.items(), key=lambda x: int(x[0])):
        label = entry['label']
        pixels = [max(-128, min(127, int(v))) for v in entry['pixels']]
        assert len(pixels) == size
        lines.append(f'static const int8_t testdata_{label}[{size}] = {{')
        for i in range(0, size, 16):
            chunk = pixels[i:i+16]
            lines.append('    ' + ', '.join(str(v) for v in chunk) + ',')
        lines.append('};')
        lines.append('')

    # testdata.digit(n) → list of 784 int8 values
    lines.append('static const int8_t *_digit_ptrs[] = {')
    for i in range(n):
        lines.append(f'    testdata_{i},')
    lines.append('};')
    lines.append('')
    lines.append(f'static mp_obj_t testdata_fn_digit(mp_obj_t n_obj) {{')
    lines.append(f'    int32_t n = mp_obj_get_int(n_obj);')
    lines.append(f'    if (n < 0 || n >= {n})')
    lines.append(f'        mp_raise_ValueError(MP_ERROR_TEXT("digit must be 0..{n-1}"));')
    lines.append(f'    const int8_t *px = _digit_ptrs[n];')
    lines.append(f'    mp_obj_t result = mp_obj_new_list({size}, NULL);')
    lines.append(f'    mp_obj_list_t *lst = MP_OBJ_TO_PTR(result);')
    lines.append(f'    for (int32_t i = 0; i < {size}; i++)')
    lines.append(f'        lst->items[i] = mp_obj_new_int((int32_t)px[i]);')
    lines.append(f'    return result;')
    lines.append(f'}}')
    lines.append(f'static MP_DEFINE_CONST_FUN_OBJ_1(testdata_fn_digit_obj, testdata_fn_digit);')
    lines.append('')
    lines.append('static const mp_rom_map_elem_t testdata_module_globals_table[] = {')
    lines.append('    { MP_ROM_QSTR(MP_QSTR___name__), MP_ROM_QSTR(MP_QSTR_testdata) },')
    lines.append('    { MP_ROM_QSTR(MP_QSTR_digit), MP_ROM_PTR(&testdata_fn_digit_obj) },')
    lines.append('};')
    lines.append('static MP_DEFINE_CONST_DICT(testdata_module_globals, testdata_module_globals_table);')
    lines.append('')
    lines.append('const mp_obj_module_t testdata_module = {')
    lines.append('    .base = { &mp_type_module },')
    lines.append('    .globals = (mp_obj_dict_t *)&testdata_module_globals,')
    lines.append('};')
    lines.append('MP_REGISTER_MODULE(MP_QSTR_testdata, testdata_module);')

    out_path = os.path.join(os.path.dirname(spec_path), 'modtestdata.c')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'Generated {out_path} ({n} digits, {size} pixels each)')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    generate(sys.argv[1])
