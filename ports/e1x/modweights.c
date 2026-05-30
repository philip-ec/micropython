#include <stdint.h>
#include "py/obj.h"
#include "py/runtime.h"

// Fabric kernel — defined in fabric_kernel.c
void matmul_int8_rq(const int8_t *a, const int8_t *b, int8_t *out,
                    int32_t M, int32_t K, int32_t N,
                    int32_t scale, int32_t shift, int32_t zero_point);

static const int8_t weights_fc1[16] = {
    64, 32, -32, -64, -32, 64, 64, -32, 32, -64, 64, 32, -64, -32, 32, 64,
};

static const int8_t weights_fc2[8] = {
    64, -64, 64, -64, 64, 64, -64, -64,
};

// weights.fc1(x, scale, shift, zero_point)
// x: list of 4 int8 values (one input vector)
// returns: list of 4 int8 values
static mp_obj_t weights_fn_fc1(size_t n_args, const mp_obj_t *args) {
    (void)n_args;
    size_t x_len; mp_obj_t *x_items;
    mp_obj_get_array(args[0], &x_len, &x_items);
    if ((int32_t)x_len != 4)
        mp_raise_ValueError(MP_ERROR_TEXT("input must have 4 elements"));
    int32_t scale = mp_obj_get_int(args[1]);
    int32_t shift = mp_obj_get_int(args[2]);
    int32_t zp    = mp_obj_get_int(args[3]);
    int8_t x_buf[4], out_buf[4];
    for (int32_t i = 0; i < 4; i++)
        x_buf[i] = (int8_t)mp_obj_get_int(x_items[i]);
    matmul_int8_rq(weights_fc1, x_buf, out_buf, 4, 4, 1, scale, shift, zp);
    mp_obj_t result = mp_obj_new_list(4, NULL);
    mp_obj_list_t *lst = MP_OBJ_TO_PTR(result);
    for (int32_t i = 0; i < 4; i++)
        lst->items[i] = mp_obj_new_int((int32_t)out_buf[i]);
    return result;
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(weights_fn_fc1_obj, 4, 4, weights_fn_fc1);

// weights.fc2(x, scale, shift, zero_point)
// x: list of 4 int8 values (one input vector)
// returns: list of 2 int8 values
static mp_obj_t weights_fn_fc2(size_t n_args, const mp_obj_t *args) {
    (void)n_args;
    size_t x_len; mp_obj_t *x_items;
    mp_obj_get_array(args[0], &x_len, &x_items);
    if ((int32_t)x_len != 4)
        mp_raise_ValueError(MP_ERROR_TEXT("input must have 4 elements"));
    int32_t scale = mp_obj_get_int(args[1]);
    int32_t shift = mp_obj_get_int(args[2]);
    int32_t zp    = mp_obj_get_int(args[3]);
    int8_t x_buf[4], out_buf[2];
    for (int32_t i = 0; i < 4; i++)
        x_buf[i] = (int8_t)mp_obj_get_int(x_items[i]);
    matmul_int8_rq(weights_fc2, x_buf, out_buf, 2, 4, 1, scale, shift, zp);
    mp_obj_t result = mp_obj_new_list(2, NULL);
    mp_obj_list_t *lst = MP_OBJ_TO_PTR(result);
    for (int32_t i = 0; i < 2; i++)
        lst->items[i] = mp_obj_new_int((int32_t)out_buf[i]);
    return result;
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(weights_fn_fc2_obj, 4, 4, weights_fn_fc2);

static const mp_rom_map_elem_t weights_module_globals_table[] = {
    { MP_ROM_QSTR(MP_QSTR___name__), MP_ROM_QSTR(MP_QSTR_weights) },
    { MP_ROM_QSTR(MP_QSTR_fc1), MP_ROM_PTR(&weights_fn_fc1_obj) },
    { MP_ROM_QSTR(MP_QSTR_fc2), MP_ROM_PTR(&weights_fn_fc2_obj) },
};
static MP_DEFINE_CONST_DICT(weights_module_globals, weights_module_globals_table);

const mp_obj_module_t weights_module = {
    .base = { &mp_type_module },
    .globals = (mp_obj_dict_t *)&weights_module_globals,
};
MP_REGISTER_MODULE(MP_QSTR_weights, weights_module);
