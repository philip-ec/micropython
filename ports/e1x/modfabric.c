#include <stdint.h>
#include "py/obj.h"
#include "py/runtime.h"

#define MAX_N 256

int32_t dot_product(const int32_t *a, const int32_t *b, int32_t n);

static mp_obj_t fabric_dot_product(mp_obj_t a_obj, mp_obj_t b_obj) {
    size_t a_len, b_len;
    mp_obj_t *a_items, *b_items;
    mp_obj_get_array(a_obj, &a_len, &a_items);
    mp_obj_get_array(b_obj, &b_len, &b_items);

    if (a_len != b_len) {
        mp_raise_ValueError(MP_ERROR_TEXT("lists must be same length"));
    }
    if (a_len > MAX_N) {
        mp_raise_ValueError(MP_ERROR_TEXT("list too long (max 256)"));
    }

    int32_t a_buf[MAX_N];
    int32_t b_buf[MAX_N];
    for (size_t i = 0; i < a_len; i++) {
        a_buf[i] = mp_obj_get_int(a_items[i]);
        b_buf[i] = mp_obj_get_int(b_items[i]);
    }

    int32_t result = dot_product(a_buf, b_buf, (int32_t)a_len);
    return mp_obj_new_int(result);
}
static MP_DEFINE_CONST_FUN_OBJ_2(fabric_dot_product_obj, fabric_dot_product);

static const mp_rom_map_elem_t fabric_module_globals_table[] = {
    { MP_ROM_QSTR(MP_QSTR___name__), MP_ROM_QSTR(MP_QSTR_fabric) },
    { MP_ROM_QSTR(MP_QSTR_dot_product), MP_ROM_PTR(&fabric_dot_product_obj) },
};
static MP_DEFINE_CONST_DICT(fabric_module_globals, fabric_module_globals_table);

const mp_obj_module_t fabric_module = {
    .base = { &mp_type_module },
    .globals = (mp_obj_dict_t *)&fabric_module_globals,
};

MP_REGISTER_MODULE(MP_QSTR_fabric, fabric_module);
