#include <stdint.h>
#include "py/obj.h"
#include "py/runtime.h"

#define MAX_N        256
#define MAX_ROWS      32
#define MAX_COLS      32
#define MAX_MATMUL    16

int32_t dot_product(const int32_t *a, const int32_t *b, int32_t n);
void relu(const int32_t *x, int32_t *out, int32_t n);
void vec_scale(const int32_t *a, int32_t scalar, int32_t *out, int32_t n);
void vec_add(const int32_t *a, const int32_t *b, int32_t *out, int32_t n);
void max_pool1d(const int32_t *signal, int32_t *out, int32_t n, int32_t w);
void matmul(const int32_t *a, const int32_t *b, int32_t *out, int32_t M, int32_t K, int32_t N);
void clip(const int32_t *a, int32_t *out, int32_t n, int32_t lo, int32_t hi);
void biquad(const int32_t *x, int32_t *y, int32_t n, int32_t b0, int32_t b1, int32_t b2, int32_t a1, int32_t a2);
void matmul_int8(const int8_t *a, const int8_t *b, int32_t *out, int32_t M, int32_t K, int32_t N);
void matvec(const int32_t *mat, const int32_t *vec, int32_t *out, int32_t rows, int32_t cols);
void mul(const int32_t *a, const int32_t *b, int32_t *out, int32_t n);
int32_t argmax(const int32_t *a, int32_t n);
void fir(const int32_t *signal, const int32_t *coeffs, int32_t *out, int32_t sig_len, int32_t n_coeffs);

// fabric.relu(x) → element-wise max(0, x)
static mp_obj_t fabric_relu(mp_obj_t x_obj) {
    size_t len;
    mp_obj_t *items;
    mp_obj_get_array(x_obj, &len, &items);
    if (len == 0 || len > MAX_N) {
        mp_raise_ValueError(MP_ERROR_TEXT("list length must be 1..256"));
    }
    int32_t in_buf[MAX_N], out_buf[MAX_N];
    for (size_t i = 0; i < len; i++) {
        in_buf[i] = mp_obj_get_int(items[i]);
    }
    relu(in_buf, out_buf, (int32_t)len);
    mp_obj_t result = mp_obj_new_list(len, NULL);
    mp_obj_list_t *result_list = MP_OBJ_TO_PTR(result);
    for (size_t i = 0; i < len; i++) {
        result_list->items[i] = mp_obj_new_int(out_buf[i]);
    }
    return result;
}
static MP_DEFINE_CONST_FUN_OBJ_1(fabric_relu_obj, fabric_relu);

// fabric.scale(a, scalar) → list where each element is a[i] * scalar
static mp_obj_t fabric_scale(mp_obj_t a_obj, mp_obj_t scalar_obj) {
    size_t a_len;
    mp_obj_t *a_items;
    mp_obj_get_array(a_obj, &a_len, &a_items);
    if (a_len == 0 || a_len > MAX_N) {
        mp_raise_ValueError(MP_ERROR_TEXT("list length must be 1..256"));
    }
    int32_t scalar = mp_obj_get_int(scalar_obj);
    int32_t a_buf[MAX_N], out_buf[MAX_N];
    for (size_t i = 0; i < a_len; i++) {
        a_buf[i] = mp_obj_get_int(a_items[i]);
    }
    vec_scale(a_buf, scalar, out_buf, (int32_t)a_len);
    mp_obj_t result = mp_obj_new_list(a_len, NULL);
    mp_obj_list_t *result_list = MP_OBJ_TO_PTR(result);
    for (size_t i = 0; i < a_len; i++) {
        result_list->items[i] = mp_obj_new_int(out_buf[i]);
    }
    return result;
}
static MP_DEFINE_CONST_FUN_OBJ_2(fabric_scale_obj, fabric_scale);

// fabric.add(a, b) → element-wise sum
static mp_obj_t fabric_add(mp_obj_t a_obj, mp_obj_t b_obj) {
    size_t a_len, b_len;
    mp_obj_t *a_items, *b_items;
    mp_obj_get_array(a_obj, &a_len, &a_items);
    mp_obj_get_array(b_obj, &b_len, &b_items);
    if (a_len != b_len) {
        mp_raise_ValueError(MP_ERROR_TEXT("lists must be same length"));
    }
    if (a_len == 0 || a_len > MAX_N) {
        mp_raise_ValueError(MP_ERROR_TEXT("list length must be 1..256"));
    }
    int32_t a_buf[MAX_N], b_buf[MAX_N], out_buf[MAX_N];
    for (size_t i = 0; i < a_len; i++) {
        a_buf[i] = mp_obj_get_int(a_items[i]);
        b_buf[i] = mp_obj_get_int(b_items[i]);
    }
    vec_add(a_buf, b_buf, out_buf, (int32_t)a_len);
    mp_obj_t result = mp_obj_new_list(a_len, NULL);
    mp_obj_list_t *result_list = MP_OBJ_TO_PTR(result);
    for (size_t i = 0; i < a_len; i++) {
        result_list->items[i] = mp_obj_new_int(out_buf[i]);
    }
    return result;
}
static MP_DEFINE_CONST_FUN_OBJ_2(fabric_add_obj, fabric_add);

// fabric.max_pool1d(signal, window_size) → non-overlapping window max
static mp_obj_t fabric_max_pool1d(mp_obj_t signal_obj, mp_obj_t window_obj) {
    size_t sig_len;
    mp_obj_t *items;
    mp_obj_get_array(signal_obj, &sig_len, &items);
    if (sig_len == 0 || sig_len > MAX_N) {
        mp_raise_ValueError(MP_ERROR_TEXT("signal length out of range"));
    }
    int32_t w = mp_obj_get_int(window_obj);
    if (w < 1 || w > (int32_t)sig_len) {
        mp_raise_ValueError(MP_ERROR_TEXT("window_size out of range"));
    }
    int32_t sig_buf[MAX_N], out_buf[MAX_N];
    for (size_t i = 0; i < sig_len; i++) {
        sig_buf[i] = mp_obj_get_int(items[i]);
    }
    int32_t out_len = (int32_t)sig_len / w;
    max_pool1d(sig_buf, out_buf, (int32_t)sig_len, w);
    mp_obj_t result = mp_obj_new_list((size_t)out_len, NULL);
    mp_obj_list_t *lst = MP_OBJ_TO_PTR(result);
    for (int32_t i = 0; i < out_len; i++) {
        lst->items[i] = mp_obj_new_int(out_buf[i]);
    }
    return result;
}
static MP_DEFINE_CONST_FUN_OBJ_2(fabric_max_pool1d_obj, fabric_max_pool1d);

// fabric.matmul(A, B) → C (int32 x int32, max 16x16)
static mp_obj_t fabric_matmul(mp_obj_t a_obj, mp_obj_t b_obj) {
    size_t M, K2, K, N;
    mp_obj_t *a_rows, *b_rows;
    mp_obj_get_array(a_obj, &M, &a_rows);
    mp_obj_get_array(b_obj, &K2, &b_rows);
    if (M == 0 || M > MAX_MATMUL) {
        mp_raise_ValueError(MP_ERROR_TEXT("A rows must be 1..16"));
    }
    if (K2 == 0 || K2 > MAX_MATMUL) {
        mp_raise_ValueError(MP_ERROR_TEXT("B rows must be 1..16"));
    }
    size_t r0_len;
    mp_obj_t *r0_items;
    mp_obj_get_array(a_rows[0], &r0_len, &r0_items);
    K = r0_len;
    if (K == 0 || K > MAX_MATMUL) {
        mp_raise_ValueError(MP_ERROR_TEXT("A cols must be 1..16"));
    }
    if (K != K2) {
        mp_raise_ValueError(MP_ERROR_TEXT("A cols must equal B rows"));
    }
    size_t b0_len;
    mp_obj_t *b0_items;
    mp_obj_get_array(b_rows[0], &b0_len, &b0_items);
    N = b0_len;
    if (N == 0 || N > MAX_MATMUL) {
        mp_raise_ValueError(MP_ERROR_TEXT("B cols must be 1..16"));
    }
    int32_t a_buf[MAX_MATMUL * MAX_MATMUL];
    int32_t b_buf[MAX_MATMUL * MAX_MATMUL];
    int32_t out_buf[MAX_MATMUL * MAX_MATMUL];
    for (size_t i = 0; i < M; i++) {
        size_t row_len;
        mp_obj_t *row_elems;
        mp_obj_get_array(a_rows[i], &row_len, &row_elems);
        if (row_len != K) {
            mp_raise_ValueError(MP_ERROR_TEXT("all A rows must be same length"));
        }
        for (size_t k = 0; k < K; k++) {
            a_buf[i * K + k] = mp_obj_get_int(row_elems[k]);
        }
    }
    for (size_t k = 0; k < K2; k++) {
        size_t row_len;
        mp_obj_t *row_elems;
        mp_obj_get_array(b_rows[k], &row_len, &row_elems);
        if (row_len != N) {
            mp_raise_ValueError(MP_ERROR_TEXT("all B rows must be same length"));
        }
        for (size_t j = 0; j < N; j++) {
            b_buf[k * N + j] = mp_obj_get_int(row_elems[j]);
        }
    }
    matmul(a_buf, b_buf, out_buf, (int32_t)M, (int32_t)K, (int32_t)N);
    mp_obj_t result = mp_obj_new_list(M, NULL);
    mp_obj_list_t *result_list = MP_OBJ_TO_PTR(result);
    for (size_t i = 0; i < M; i++) {
        mp_obj_t row = mp_obj_new_list(N, NULL);
        mp_obj_list_t *row_list = MP_OBJ_TO_PTR(row);
        for (size_t j = 0; j < N; j++) {
            row_list->items[j] = mp_obj_new_int(out_buf[i * N + j]);
        }
        result_list->items[i] = row;
    }
    return result;
}
static MP_DEFINE_CONST_FUN_OBJ_2(fabric_matmul_obj, fabric_matmul);

// fabric.clip(a, lo, hi) → element-wise clamp to [lo, hi]
static mp_obj_t fabric_clip(size_t n_args, const mp_obj_t *args) {
    (void)n_args;
    size_t a_len;
    mp_obj_t *a_items;
    mp_obj_get_array(args[0], &a_len, &a_items);
    if (a_len == 0 || a_len > MAX_N) {
        mp_raise_ValueError(MP_ERROR_TEXT("list length must be 1..256"));
    }
    int32_t lo = mp_obj_get_int(args[1]);
    int32_t hi = mp_obj_get_int(args[2]);
    if (lo > hi) {
        mp_raise_ValueError(MP_ERROR_TEXT("lo must be <= hi"));
    }
    int32_t a_buf[MAX_N], out_buf[MAX_N];
    for (size_t i = 0; i < a_len; i++) {
        a_buf[i] = mp_obj_get_int(a_items[i]);
    }
    clip(a_buf, out_buf, (int32_t)a_len, lo, hi);
    mp_obj_t result = mp_obj_new_list(a_len, NULL);
    mp_obj_list_t *result_list = MP_OBJ_TO_PTR(result);
    for (size_t i = 0; i < a_len; i++) {
        result_list->items[i] = mp_obj_new_int(out_buf[i]);
    }
    return result;
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(fabric_clip_obj, 3, 3, fabric_clip);

// fabric.biquad(signal, coeffs)
// signal: list of ints (raw samples), max 256
// coeffs: list of 5 ints [b0, b1, b2, a1, a2] in Q15 (multiply floats by 32768)
// returns: list of filtered ints
static mp_obj_t fabric_biquad(mp_obj_t signal_obj, mp_obj_t coeffs_obj) {
    size_t sig_len, n_coeffs;
    mp_obj_t *sig_items, *coeff_items;
    mp_obj_get_array(signal_obj, &sig_len, &sig_items);
    mp_obj_get_array(coeffs_obj, &n_coeffs, &coeff_items);

    if (n_coeffs != 5) {
        mp_raise_ValueError(MP_ERROR_TEXT("coeffs must be [b0,b1,b2,a1,a2]"));
    }
    if (sig_len == 0 || sig_len > MAX_N) {
        mp_raise_ValueError(MP_ERROR_TEXT("signal length must be 1..256"));
    }

    int32_t sig_buf[MAX_N], out_buf[MAX_N];
    for (size_t i = 0; i < sig_len; i++) {
        sig_buf[i] = mp_obj_get_int(sig_items[i]);
    }

    int32_t b0 = mp_obj_get_int(coeff_items[0]);
    int32_t b1 = mp_obj_get_int(coeff_items[1]);
    int32_t b2 = mp_obj_get_int(coeff_items[2]);
    int32_t a1 = mp_obj_get_int(coeff_items[3]);
    int32_t a2 = mp_obj_get_int(coeff_items[4]);

    biquad(sig_buf, out_buf, (int32_t)sig_len, b0, b1, b2, a1, a2);

    mp_obj_t result = mp_obj_new_list(sig_len, NULL);
    mp_obj_list_t *result_list = MP_OBJ_TO_PTR(result);
    for (size_t i = 0; i < sig_len; i++) {
        result_list->items[i] = mp_obj_new_int(out_buf[i]);
    }
    return result;
}
static MP_DEFINE_CONST_FUN_OBJ_2(fabric_biquad_obj, fabric_biquad);

// fabric.matmul_int8(A, B)
// A: list of rows (int8 values, -128..127), shape MxK, max 16x16
// B: list of rows (int8 values, -128..127), shape KxN, max 16x16
// returns: list of rows (int32 accumulators), shape MxN
static mp_obj_t fabric_matmul_int8(mp_obj_t a_obj, mp_obj_t b_obj) {
    size_t M, K, K2, N;
    mp_obj_t *a_rows, *b_rows;

    mp_obj_get_array(a_obj, &M, &a_rows);
    mp_obj_get_array(b_obj, &K2, &b_rows);

    if (M == 0 || M > MAX_MATMUL) {
        mp_raise_ValueError(MP_ERROR_TEXT("A rows must be 1..16"));
    }

    size_t r0_len;
    mp_obj_t *r0_items;
    mp_obj_get_array(a_rows[0], &r0_len, &r0_items);
    K = r0_len;

    if (K == 0 || K > MAX_MATMUL) {
        mp_raise_ValueError(MP_ERROR_TEXT("A cols must be 1..16"));
    }
    if (K2 != K) {
        mp_raise_ValueError(MP_ERROR_TEXT("A cols must equal B rows"));
    }

    size_t b0_len;
    mp_obj_t *b0_items;
    mp_obj_get_array(b_rows[0], &b0_len, &b0_items);
    N = b0_len;

    if (N == 0 || N > MAX_MATMUL) {
        mp_raise_ValueError(MP_ERROR_TEXT("B cols must be 1..16"));
    }

    int8_t  a_buf[MAX_MATMUL * MAX_MATMUL];
    int8_t  b_buf[MAX_MATMUL * MAX_MATMUL];
    int32_t out_buf[MAX_MATMUL * MAX_MATMUL];

    for (size_t i = 0; i < M; i++) {
        size_t row_len;
        mp_obj_t *row_elems;
        mp_obj_get_array(a_rows[i], &row_len, &row_elems);
        if (row_len != K) {
            mp_raise_ValueError(MP_ERROR_TEXT("all A rows must be same length"));
        }
        for (size_t k = 0; k < K; k++) {
            a_buf[i * K + k] = (int8_t)mp_obj_get_int(row_elems[k]);
        }
    }
    for (size_t k = 0; k < K2; k++) {
        size_t row_len;
        mp_obj_t *row_elems;
        mp_obj_get_array(b_rows[k], &row_len, &row_elems);
        if (row_len != N) {
            mp_raise_ValueError(MP_ERROR_TEXT("all B rows must be same length"));
        }
        for (size_t j = 0; j < N; j++) {
            b_buf[k * N + j] = (int8_t)mp_obj_get_int(row_elems[j]);
        }
    }

    matmul_int8(a_buf, b_buf, out_buf, (int32_t)M, (int32_t)K, (int32_t)N);

    mp_obj_t result = mp_obj_new_list(M, NULL);
    mp_obj_list_t *result_list = MP_OBJ_TO_PTR(result);
    for (size_t i = 0; i < M; i++) {
        mp_obj_t row = mp_obj_new_list(N, NULL);
        mp_obj_list_t *row_list = MP_OBJ_TO_PTR(row);
        for (size_t j = 0; j < N; j++) {
            row_list->items[j] = mp_obj_new_int(out_buf[i * N + j]);
        }
        result_list->items[i] = row;
    }
    return result;
}
static MP_DEFINE_CONST_FUN_OBJ_2(fabric_matmul_int8_obj, fabric_matmul_int8);

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

// fabric.mul(a, b) → element-wise multiply
static mp_obj_t fabric_mul(mp_obj_t a_obj, mp_obj_t b_obj) {
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

    int32_t a_buf[MAX_N], b_buf[MAX_N], out_buf[MAX_N];
    for (size_t i = 0; i < a_len; i++) {
        a_buf[i] = mp_obj_get_int(a_items[i]);
        b_buf[i] = mp_obj_get_int(b_items[i]);
    }

    mul(a_buf, b_buf, out_buf, (int32_t)a_len);

    mp_obj_t result = mp_obj_new_list(a_len, NULL);
    mp_obj_list_t *result_list = MP_OBJ_TO_PTR(result);
    for (size_t i = 0; i < a_len; i++) {
        result_list->items[i] = mp_obj_new_int(out_buf[i]);
    }
    return result;
}
static MP_DEFINE_CONST_FUN_OBJ_2(fabric_mul_obj, fabric_mul);

// fabric.argmax(scores) → index of maximum value
static mp_obj_t fabric_argmax(mp_obj_t a_obj) {
    size_t len;
    mp_obj_t *items;
    mp_obj_get_array(a_obj, &len, &items);

    if (len == 0) {
        mp_raise_ValueError(MP_ERROR_TEXT("list is empty"));
    }
    if (len > MAX_N) {
        mp_raise_ValueError(MP_ERROR_TEXT("list too long (max 256)"));
    }

    int32_t buf[MAX_N];
    for (size_t i = 0; i < len; i++) {
        buf[i] = mp_obj_get_int(items[i]);
    }

    int32_t idx = argmax(buf, (int32_t)len);
    return mp_obj_new_int(idx);
}
static MP_DEFINE_CONST_FUN_OBJ_1(fabric_argmax_obj, fabric_argmax);

// fabric.fir(signal, coeffs)
// signal: list of ints, coeffs: list of ints (taps)
// returns: list of ints, length == len(signal) - len(coeffs) + 1
static mp_obj_t fabric_fir(mp_obj_t signal_obj, mp_obj_t coeffs_obj) {
    size_t sig_len, n_coeffs;
    mp_obj_t *sig_items, *coeff_items;
    mp_obj_get_array(signal_obj, &sig_len, &sig_items);
    mp_obj_get_array(coeffs_obj, &n_coeffs, &coeff_items);

    if (n_coeffs == 0 || n_coeffs > MAX_N) {
        mp_raise_ValueError(MP_ERROR_TEXT("coeffs length must be 1..256"));
    }
    if (sig_len < n_coeffs) {
        mp_raise_ValueError(MP_ERROR_TEXT("signal shorter than coeffs"));
    }
    if (sig_len > MAX_N) {
        mp_raise_ValueError(MP_ERROR_TEXT("signal too long (max 256)"));
    }

    int32_t sig_buf[MAX_N];
    int32_t coeff_buf[MAX_N];
    int32_t out_buf[MAX_N];

    for (size_t i = 0; i < sig_len; i++) {
        sig_buf[i] = mp_obj_get_int(sig_items[i]);
    }
    for (size_t i = 0; i < n_coeffs; i++) {
        coeff_buf[i] = mp_obj_get_int(coeff_items[i]);
    }

    int32_t out_len = (int32_t)(sig_len - n_coeffs + 1);
    fir(sig_buf, coeff_buf, out_buf, (int32_t)sig_len, (int32_t)n_coeffs);

    mp_obj_t result = mp_obj_new_list((size_t)out_len, NULL);
    mp_obj_list_t *result_list = MP_OBJ_TO_PTR(result);
    for (int32_t i = 0; i < out_len; i++) {
        result_list->items[i] = mp_obj_new_int(out_buf[i]);
    }
    return result;
}
static MP_DEFINE_CONST_FUN_OBJ_2(fabric_fir_obj, fabric_fir);

static const mp_rom_map_elem_t fabric_module_globals_table[] = {
    { MP_ROM_QSTR(MP_QSTR___name__), MP_ROM_QSTR(MP_QSTR_fabric) },
    { MP_ROM_QSTR(MP_QSTR_relu), MP_ROM_PTR(&fabric_relu_obj) },
    { MP_ROM_QSTR(MP_QSTR_scale), MP_ROM_PTR(&fabric_scale_obj) },
    { MP_ROM_QSTR(MP_QSTR_add), MP_ROM_PTR(&fabric_add_obj) },
    { MP_ROM_QSTR(MP_QSTR_max_pool1d), MP_ROM_PTR(&fabric_max_pool1d_obj) },
    { MP_ROM_QSTR(MP_QSTR_matmul), MP_ROM_PTR(&fabric_matmul_obj) },
    { MP_ROM_QSTR(MP_QSTR_clip), MP_ROM_PTR(&fabric_clip_obj) },
    { MP_ROM_QSTR(MP_QSTR_biquad), MP_ROM_PTR(&fabric_biquad_obj) },
    { MP_ROM_QSTR(MP_QSTR_matmul_int8), MP_ROM_PTR(&fabric_matmul_int8_obj) },
    { MP_ROM_QSTR(MP_QSTR_dot_product), MP_ROM_PTR(&fabric_dot_product_obj) },
    { MP_ROM_QSTR(MP_QSTR_matvec), MP_ROM_PTR(&fabric_matvec_obj) },
    { MP_ROM_QSTR(MP_QSTR_fir), MP_ROM_PTR(&fabric_fir_obj) },
    { MP_ROM_QSTR(MP_QSTR_argmax), MP_ROM_PTR(&fabric_argmax_obj) },
    { MP_ROM_QSTR(MP_QSTR_mul), MP_ROM_PTR(&fabric_mul_obj) },
};
static MP_DEFINE_CONST_DICT(fabric_module_globals, fabric_module_globals_table);

const mp_obj_module_t fabric_module = {
    .base = { &mp_type_module },
    .globals = (mp_obj_dict_t *)&fabric_module_globals,
};

MP_REGISTER_MODULE(MP_QSTR_fabric, fabric_module);
