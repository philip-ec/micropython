#include <stdint.h>
#include "py/obj.h"
#include "py/runtime.h"

#define MAX_N    256
#define MAX_ROWS  32
#define MAX_COLS  32

int32_t dot_product(const int32_t *a, const int32_t *b, int32_t n);
void matvec(const int32_t *mat, const int32_t *vec, int32_t *out, int32_t rows, int32_t cols);

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

// fabric.matvec(matrix, vector)
// matrix: list of rows (each row a list of ints), max 32x32
// vector: list of ints, length == num cols
// returns: list of ints, length == num rows
static mp_obj_t fabric_matvec(mp_obj_t mat_obj, mp_obj_t vec_obj) {
    size_t rows, cols, vec_len;
    mp_obj_t *row_items, *vec_items;

    mp_obj_get_array(mat_obj, &rows, &row_items);
    mp_obj_get_array(vec_obj, &vec_len, &vec_items);

    if (rows == 0 || rows > MAX_ROWS) {
        mp_raise_ValueError(MP_ERROR_TEXT("matrix rows must be 1..32"));
    }

    // infer cols from first row
    size_t r0_len;
    mp_obj_t *r0_items;
    mp_obj_get_array(row_items[0], &r0_len, &r0_items);
    cols = r0_len;

    if (cols == 0 || cols > MAX_COLS) {
        mp_raise_ValueError(MP_ERROR_TEXT("matrix cols must be 1..32"));
    }
    if (vec_len != cols) {
        mp_raise_ValueError(MP_ERROR_TEXT("vector length must match matrix cols"));
    }

    // unbox matrix (row-major) and vector
    int32_t mat_buf[MAX_ROWS * MAX_COLS];
    int32_t vec_buf[MAX_COLS];
    int32_t out_buf[MAX_ROWS];

    for (size_t i = 0; i < rows; i++) {
        size_t row_len;
        mp_obj_t *row_elems;
        mp_obj_get_array(row_items[i], &row_len, &row_elems);
        if (row_len != cols) {
            mp_raise_ValueError(MP_ERROR_TEXT("all rows must be same length"));
        }
        for (size_t j = 0; j < cols; j++) {
            mat_buf[i * cols + j] = mp_obj_get_int(row_elems[j]);
        }
    }
    for (size_t j = 0; j < cols; j++) {
        vec_buf[j] = mp_obj_get_int(vec_items[j]);
    }

    matvec(mat_buf, vec_buf, out_buf, (int32_t)rows, (int32_t)cols);

    // box result as Python list
    mp_obj_t result = mp_obj_new_list((size_t)rows, NULL);
    mp_obj_list_t *result_list = MP_OBJ_TO_PTR(result);
    for (size_t i = 0; i < rows; i++) {
        result_list->items[i] = mp_obj_new_int(out_buf[i]);
    }
    return result;
}
static MP_DEFINE_CONST_FUN_OBJ_2(fabric_matvec_obj, fabric_matvec);

static const mp_rom_map_elem_t fabric_module_globals_table[] = {
    { MP_ROM_QSTR(MP_QSTR___name__), MP_ROM_QSTR(MP_QSTR_fabric) },
    { MP_ROM_QSTR(MP_QSTR_dot_product), MP_ROM_PTR(&fabric_dot_product_obj) },
    { MP_ROM_QSTR(MP_QSTR_matvec), MP_ROM_PTR(&fabric_matvec_obj) },
};
static MP_DEFINE_CONST_DICT(fabric_module_globals, fabric_module_globals_table);

const mp_obj_module_t fabric_module = {
    .base = { &mp_type_module },
    .globals = (mp_obj_dict_t *)&fabric_module_globals,
};

MP_REGISTER_MODULE(MP_QSTR_fabric, fabric_module);
