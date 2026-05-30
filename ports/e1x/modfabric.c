#include <stdint.h>
#include "py/obj.h"
#include "py/runtime.h"
#include <eff/mtimer.h>
#include "fft.h"

#define MAX_N        256
#define MAX_ROWS      32
#define MAX_COLS      32
#define MAX_MATMUL    16

int32_t dot_product(const int32_t *a, const int32_t *b, int32_t n);
void matmul_int8_rq(const int8_t *a, const int8_t *b, int8_t *out, int32_t M, int32_t K, int32_t N, int32_t scale, int32_t shift, int32_t zero_point);
void conv2d_int8_rq(const int8_t *input, const int8_t *kernel, int8_t *out, int32_t in_h, int32_t in_w, int32_t k_h, int32_t k_w, int32_t scale, int32_t shift, int32_t zero_point);
void pointwise_conv_rq(const int8_t *input, const int8_t *weights, int8_t *out, int32_t in_ch, int32_t out_ch, int32_t scale, int32_t shift, int32_t zero_point);
void requantize(const int32_t *acc, int8_t *out, int32_t n, int32_t scale, int32_t shift, int32_t zero_point);
int32_t sum_squares(const int32_t *a, int32_t n);
void l2_norm(const int32_t *a, int32_t *out, int32_t n, int32_t scale);
void biquad_cascade(const int32_t *signal, const int32_t *coeffs, int32_t *out, int32_t *buf, int32_t n, int32_t n_sections);
void conv2d_int8(const int8_t *input, const int8_t *kernel, int32_t *out, int32_t in_h, int32_t in_w, int32_t k_h, int32_t k_w);
void conv1d(const int32_t *signal, const int32_t *kernels, int32_t *out, int32_t in_ch, int32_t out_ch, int32_t k_len, int32_t sig_len);
void avg_pool1d(const int32_t *signal, int32_t *out, int32_t n, int32_t w);
void pointwise_conv(const int8_t *input, const int8_t *weights, int32_t *out, int32_t in_ch, int32_t out_ch);
void vec_threshold(const int32_t *a, int32_t thresh, int32_t *out, int32_t n);
void vec_vmax(const int32_t *a, const int32_t *b, int32_t *out, int32_t n);
void vec_vmin(const int32_t *a, const int32_t *b, int32_t *out, int32_t n);
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

#define MAX_CONV1D_CH 8

// fabric.matmul_int8_rq(A, B, scale, shift, zero_point) -> list of lists of int8
// Fused matmul_int8 + requantize — no Python round-trip for the int32 accumulators.
static mp_obj_t fabric_matmul_int8_rq(size_t n_args, const mp_obj_t *args) {
    (void)n_args;
    size_t M, K2, K, N;
    mp_obj_t *a_rows, *b_rows;
    mp_obj_get_array(args[0], &M, &a_rows);
    mp_obj_get_array(args[1], &K2, &b_rows);
    if (M == 0 || M > MAX_MATMUL)  mp_raise_ValueError(MP_ERROR_TEXT("A rows must be 1..16"));
    if (K2 == 0 || K2 > MAX_MATMUL) mp_raise_ValueError(MP_ERROR_TEXT("B rows must be 1..16"));
    size_t r0_len; mp_obj_t *r0_items;
    mp_obj_get_array(a_rows[0], &r0_len, &r0_items);
    K = r0_len;
    if (K == 0 || K > MAX_MATMUL || K != K2) mp_raise_ValueError(MP_ERROR_TEXT("inner dims must match, 1..16"));
    size_t b0_len; mp_obj_t *b0_items;
    mp_obj_get_array(b_rows[0], &b0_len, &b0_items);
    N = b0_len;
    if (N == 0 || N > MAX_MATMUL) mp_raise_ValueError(MP_ERROR_TEXT("B cols must be 1..16"));
    int32_t scale = mp_obj_get_int(args[2]);
    int32_t shift = mp_obj_get_int(args[3]);
    int32_t zp    = mp_obj_get_int(args[4]);
    if (shift < 0 || shift > 31) mp_raise_ValueError(MP_ERROR_TEXT("shift must be 0..31"));
    int8_t a_buf[MAX_MATMUL*MAX_MATMUL], b_buf[MAX_MATMUL*MAX_MATMUL], out_buf[MAX_MATMUL*MAX_MATMUL];
    for (size_t i = 0; i < M; i++) {
        size_t row_len; mp_obj_t *row_elems;
        mp_obj_get_array(a_rows[i], &row_len, &row_elems);
        if (row_len != K) mp_raise_ValueError(MP_ERROR_TEXT("all A rows must be same length"));
        for (size_t k = 0; k < K; k++) a_buf[i*K+k] = (int8_t)mp_obj_get_int(row_elems[k]);
    }
    for (size_t k = 0; k < K2; k++) {
        size_t row_len; mp_obj_t *row_elems;
        mp_obj_get_array(b_rows[k], &row_len, &row_elems);
        if (row_len != N) mp_raise_ValueError(MP_ERROR_TEXT("all B rows must be same length"));
        for (size_t j = 0; j < N; j++) b_buf[k*N+j] = (int8_t)mp_obj_get_int(row_elems[j]);
    }
    matmul_int8_rq(a_buf, b_buf, out_buf, (int32_t)M, (int32_t)K, (int32_t)N, scale, shift, zp);
    mp_obj_t result = mp_obj_new_list(M, NULL);
    mp_obj_list_t *result_list = MP_OBJ_TO_PTR(result);
    for (size_t i = 0; i < M; i++) {
        mp_obj_t row = mp_obj_new_list(N, NULL);
        mp_obj_list_t *row_list = MP_OBJ_TO_PTR(row);
        for (size_t j = 0; j < N; j++) row_list->items[j] = mp_obj_new_int((int32_t)out_buf[i*N+j]);
        result_list->items[i] = row;
    }
    return result;
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(fabric_matmul_int8_rq_obj, 5, 5, fabric_matmul_int8_rq);

// fabric.conv2d_int8_rq(input, kernel, in_h, in_w, k_h, k_w, scale, shift, zero_point) -> flat list of int8
static mp_obj_t fabric_conv2d_int8_rq(size_t n_args, const mp_obj_t *args) {
    (void)n_args;
    size_t inp_len, kern_len;
    mp_obj_t *inp_items, *kern_items;
    mp_obj_get_array(args[0], &inp_len, &inp_items);
    mp_obj_get_array(args[1], &kern_len, &kern_items);
    int32_t in_h  = mp_obj_get_int(args[2]);
    int32_t in_w  = mp_obj_get_int(args[3]);
    int32_t k_h   = mp_obj_get_int(args[4]);
    int32_t k_w   = mp_obj_get_int(args[5]);
    int32_t scale = mp_obj_get_int(args[6]);
    int32_t shift = mp_obj_get_int(args[7]);
    int32_t zp    = mp_obj_get_int(args[8]);
    if (in_h < 1 || in_w < 1 || in_h*in_w > MAX_N) mp_raise_ValueError(MP_ERROR_TEXT("input must be <= 256 elements"));
    if (k_h < 1 || k_w < 1 || k_h > 8 || k_w > 8)  mp_raise_ValueError(MP_ERROR_TEXT("kernel dims must be 1..8"));
    if (k_h > in_h || k_w > in_w)                    mp_raise_ValueError(MP_ERROR_TEXT("kernel exceeds input"));
    if (inp_len != (size_t)(in_h*in_w) || kern_len != (size_t)(k_h*k_w)) mp_raise_ValueError(MP_ERROR_TEXT("length mismatch"));
    if (shift < 0 || shift > 31)                      mp_raise_ValueError(MP_ERROR_TEXT("shift must be 0..31"));
    int8_t inp_buf[MAX_N], kern_buf[64], out_buf[MAX_N];
    for (size_t i = 0; i < inp_len;  i++) inp_buf[i]  = (int8_t)mp_obj_get_int(inp_items[i]);
    for (size_t i = 0; i < kern_len; i++) kern_buf[i] = (int8_t)mp_obj_get_int(kern_items[i]);
    int32_t out_len = (in_h-k_h+1) * (in_w-k_w+1);
    conv2d_int8_rq(inp_buf, kern_buf, out_buf, in_h, in_w, k_h, k_w, scale, shift, zp);
    mp_obj_t result = mp_obj_new_list((size_t)out_len, NULL);
    mp_obj_list_t *result_list = MP_OBJ_TO_PTR(result);
    for (int32_t i = 0; i < out_len; i++) result_list->items[i] = mp_obj_new_int((int32_t)out_buf[i]);
    return result;
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(fabric_conv2d_int8_rq_obj, 9, 9, fabric_conv2d_int8_rq);

// fabric.pointwise_conv_rq(input, weights, in_ch, out_ch, scale, shift, zero_point) -> list of int8
static mp_obj_t fabric_pointwise_conv_rq(size_t n_args, const mp_obj_t *args) {
    (void)n_args;
    size_t inp_len, w_len;
    mp_obj_t *inp_items, *w_items;
    mp_obj_get_array(args[0], &inp_len, &inp_items);
    mp_obj_get_array(args[1], &w_len,   &w_items);
    int32_t in_ch  = mp_obj_get_int(args[2]);
    int32_t out_ch = mp_obj_get_int(args[3]);
    int32_t scale  = mp_obj_get_int(args[4]);
    int32_t shift  = mp_obj_get_int(args[5]);
    int32_t zp     = mp_obj_get_int(args[6]);
    if (in_ch < 1 || in_ch > MAX_MATMUL)   mp_raise_ValueError(MP_ERROR_TEXT("in_ch must be 1..16"));
    if (out_ch < 1 || out_ch > MAX_MATMUL)  mp_raise_ValueError(MP_ERROR_TEXT("out_ch must be 1..16"));
    if ((int32_t)inp_len != in_ch)           mp_raise_ValueError(MP_ERROR_TEXT("input length must equal in_ch"));
    if ((int32_t)w_len != out_ch*in_ch)      mp_raise_ValueError(MP_ERROR_TEXT("weights length must equal out_ch*in_ch"));
    if (shift < 0 || shift > 31)             mp_raise_ValueError(MP_ERROR_TEXT("shift must be 0..31"));
    int8_t inp_buf[MAX_MATMUL], w_buf[MAX_MATMUL*MAX_MATMUL], out_buf[MAX_MATMUL];
    for (int32_t i = 0; i < in_ch;        i++) inp_buf[i] = (int8_t)mp_obj_get_int(inp_items[i]);
    for (int32_t i = 0; i < out_ch*in_ch; i++) w_buf[i]   = (int8_t)mp_obj_get_int(w_items[i]);
    pointwise_conv_rq(inp_buf, w_buf, out_buf, in_ch, out_ch, scale, shift, zp);
    mp_obj_t result = mp_obj_new_list((size_t)out_ch, NULL);
    mp_obj_list_t *result_list = MP_OBJ_TO_PTR(result);
    for (int32_t i = 0; i < out_ch; i++) result_list->items[i] = mp_obj_new_int((int32_t)out_buf[i]);
    return result;
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(fabric_pointwise_conv_rq_obj, 7, 7, fabric_pointwise_conv_rq);

// fabric.requantize(acc, scale, shift, zero_point) -> list of int8 as Python ints
static mp_obj_t fabric_requantize(size_t n_args, const mp_obj_t *args) {
    (void)n_args;
    size_t acc_len;
    mp_obj_t *acc_items;
    mp_obj_get_array(args[0], &acc_len, &acc_items);
    if (acc_len == 0 || acc_len > MAX_N) {
        mp_raise_ValueError(MP_ERROR_TEXT("acc length must be 1..256"));
    }
    int32_t scale      = mp_obj_get_int(args[1]);
    int32_t shift      = mp_obj_get_int(args[2]);
    int32_t zero_point = mp_obj_get_int(args[3]);
    if (shift < 0 || shift > 31) {
        mp_raise_ValueError(MP_ERROR_TEXT("shift must be 0..31"));
    }
    int32_t acc_buf[MAX_N];
    int8_t  out_buf[MAX_N];
    for (size_t i = 0; i < acc_len; i++) {
        acc_buf[i] = mp_obj_get_int(acc_items[i]);
    }
    requantize(acc_buf, out_buf, (int32_t)acc_len, scale, shift, zero_point);
    mp_obj_t result = mp_obj_new_list(acc_len, NULL);
    mp_obj_list_t *result_list = MP_OBJ_TO_PTR(result);
    for (size_t i = 0; i < acc_len; i++) {
        result_list->items[i] = mp_obj_new_int((int32_t)out_buf[i]);
    }
    return result;
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(fabric_requantize_obj, 4, 4, fabric_requantize);

// fabric.sum_squares(a) -> int
static mp_obj_t fabric_sum_squares(mp_obj_t a_obj) {
    size_t len;
    mp_obj_t *items;
    mp_obj_get_array(a_obj, &len, &items);
    if (len == 0 || len > MAX_N) {
        mp_raise_ValueError(MP_ERROR_TEXT("list length must be 1..256"));
    }
    int32_t buf[MAX_N];
    for (size_t i = 0; i < len; i++) {
        buf[i] = mp_obj_get_int(items[i]);
    }
    return mp_obj_new_int(sum_squares(buf, (int32_t)len));
}
static MP_DEFINE_CONST_FUN_OBJ_1(fabric_sum_squares_obj, fabric_sum_squares);

// fabric.l2_norm(a, scale) -> list
static mp_obj_t fabric_l2_norm(mp_obj_t a_obj, mp_obj_t scale_obj) {
    size_t len;
    mp_obj_t *items;
    mp_obj_get_array(a_obj, &len, &items);
    if (len == 0 || len > MAX_N) {
        mp_raise_ValueError(MP_ERROR_TEXT("list length must be 1..256"));
    }
    int32_t scale = mp_obj_get_int(scale_obj);
    int32_t in_buf[MAX_N], out_buf[MAX_N];
    for (size_t i = 0; i < len; i++) {
        in_buf[i] = mp_obj_get_int(items[i]);
    }
    if (sum_squares(in_buf, (int32_t)len) == 0) {
        mp_raise_ValueError(MP_ERROR_TEXT("l2_norm: input vector is all-zero"));
    }
    l2_norm(in_buf, out_buf, (int32_t)len, scale);
    mp_obj_t result = mp_obj_new_list(len, NULL);
    mp_obj_list_t *result_list = MP_OBJ_TO_PTR(result);
    for (size_t i = 0; i < len; i++) {
        result_list->items[i] = mp_obj_new_int(out_buf[i]);
    }
    return result;
}
static MP_DEFINE_CONST_FUN_OBJ_2(fabric_l2_norm_obj, fabric_l2_norm);

// fabric.biquad_cascade(signal, sections) -> list
static mp_obj_t fabric_biquad_cascade(mp_obj_t signal_obj, mp_obj_t sections_obj) {
    size_t sig_len;
    mp_obj_t *sig_items;
    mp_obj_get_array(signal_obj, &sig_len, &sig_items);
    if (sig_len == 0 || sig_len > MAX_N) {
        mp_raise_ValueError(MP_ERROR_TEXT("signal length must be 1..256"));
    }
    size_t n_sections;
    mp_obj_t *section_items;
    mp_obj_get_array(sections_obj, &n_sections, &section_items);
    if (n_sections == 0 || n_sections > 8) {
        mp_raise_ValueError(MP_ERROR_TEXT("sections must be 1..8"));
    }
    int32_t sig_buf[MAX_N], out_buf[MAX_N], scratch[MAX_N];
    int32_t coeffs[8 * 5];
    for (size_t i = 0; i < sig_len; i++) {
        sig_buf[i] = mp_obj_get_int(sig_items[i]);
    }
    for (size_t s = 0; s < n_sections; s++) {
        size_t coeff_len;
        mp_obj_t *coeff_items;
        mp_obj_get_array(section_items[s], &coeff_len, &coeff_items);
        if (coeff_len != 5) {
            mp_raise_ValueError(MP_ERROR_TEXT("each section must be [b0,b1,b2,a1,a2]"));
        }
        for (int k = 0; k < 5; k++) {
            coeffs[s*5+k] = mp_obj_get_int(coeff_items[k]);
        }
    }
    biquad_cascade(sig_buf, coeffs, out_buf, scratch, (int32_t)sig_len, (int32_t)n_sections);
    mp_obj_t result = mp_obj_new_list(sig_len, NULL);
    mp_obj_list_t *result_list = MP_OBJ_TO_PTR(result);
    for (size_t i = 0; i < sig_len; i++) {
        result_list->items[i] = mp_obj_new_int(out_buf[i]);
    }
    return result;
}
static MP_DEFINE_CONST_FUN_OBJ_2(fabric_biquad_cascade_obj, fabric_biquad_cascade);

// fabric.conv2d_int8(input, kernel, in_h, in_w, k_h, k_w) -> flat list of int32
static mp_obj_t fabric_conv2d_int8(size_t n_args, const mp_obj_t *args) {
    (void)n_args;
    size_t inp_len, kern_len;
    mp_obj_t *inp_items, *kern_items;
    mp_obj_get_array(args[0], &inp_len, &inp_items);
    mp_obj_get_array(args[1], &kern_len, &kern_items);
    int32_t in_h = mp_obj_get_int(args[2]);
    int32_t in_w = mp_obj_get_int(args[3]);
    int32_t k_h  = mp_obj_get_int(args[4]);
    int32_t k_w  = mp_obj_get_int(args[5]);
    if (in_h < 1 || in_w < 1 || in_h*in_w > MAX_N) {
        mp_raise_ValueError(MP_ERROR_TEXT("input must be <= 256 total elements"));
    }
    if (k_h < 1 || k_w < 1 || k_h > 8 || k_w > 8) {
        mp_raise_ValueError(MP_ERROR_TEXT("kernel dims must be 1..8"));
    }
    if (k_h > in_h || k_w > in_w) {
        mp_raise_ValueError(MP_ERROR_TEXT("kernel must not exceed input dimensions"));
    }
    if (inp_len != (size_t)(in_h*in_w) || kern_len != (size_t)(k_h*k_w)) {
        mp_raise_ValueError(MP_ERROR_TEXT("input/kernel length mismatch"));
    }
    int8_t  inp_buf[MAX_N], kern_buf[64];
    int32_t out_buf[MAX_N];
    for (size_t i = 0; i < inp_len;  i++) inp_buf[i]  = (int8_t)mp_obj_get_int(inp_items[i]);
    for (size_t i = 0; i < kern_len; i++) kern_buf[i] = (int8_t)mp_obj_get_int(kern_items[i]);
    int32_t out_len = (in_h-k_h+1) * (in_w-k_w+1);
    conv2d_int8(inp_buf, kern_buf, out_buf, in_h, in_w, k_h, k_w);
    mp_obj_t result = mp_obj_new_list((size_t)out_len, NULL);
    mp_obj_list_t *result_list = MP_OBJ_TO_PTR(result);
    for (int32_t i = 0; i < out_len; i++) {
        result_list->items[i] = mp_obj_new_int(out_buf[i]);
    }
    return result;
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(fabric_conv2d_int8_obj, 6, 6, fabric_conv2d_int8);

// fabric.conv1d(signal, kernels, in_ch, out_ch, k_len) -> flat list
static mp_obj_t fabric_conv1d(size_t n_args, const mp_obj_t *args) {
    (void)n_args;
    size_t sig_total, kern_total;
    mp_obj_t *sig_items, *kern_items;
    mp_obj_get_array(args[0], &sig_total, &sig_items);
    mp_obj_get_array(args[1], &kern_total, &kern_items);
    int32_t in_ch  = mp_obj_get_int(args[2]);
    int32_t out_ch = mp_obj_get_int(args[3]);
    int32_t k_len  = mp_obj_get_int(args[4]);
    if (in_ch < 1 || in_ch > MAX_CONV1D_CH)  mp_raise_ValueError(MP_ERROR_TEXT("in_ch must be 1..8"));
    if (out_ch < 1 || out_ch > MAX_CONV1D_CH) mp_raise_ValueError(MP_ERROR_TEXT("out_ch must be 1..8"));
    if (k_len < 1)                             mp_raise_ValueError(MP_ERROR_TEXT("k_len must be >= 1"));
    if (sig_total == 0 || sig_total > MAX_N)   mp_raise_ValueError(MP_ERROR_TEXT("signal total must be 1..256"));
    if (sig_total % (size_t)in_ch != 0)        mp_raise_ValueError(MP_ERROR_TEXT("signal length not divisible by in_ch"));
    int32_t sig_len = (int32_t)(sig_total / (size_t)in_ch);
    if (sig_len < k_len)                       mp_raise_ValueError(MP_ERROR_TEXT("signal shorter than kernel"));
    if (kern_total != (size_t)(out_ch*in_ch*k_len)) mp_raise_ValueError(MP_ERROR_TEXT("kernels length must be out_ch*in_ch*k_len"));
    if (kern_total > MAX_N)                    mp_raise_ValueError(MP_ERROR_TEXT("kernels total must be <= 256"));
    int32_t out_len   = sig_len - k_len + 1;
    int32_t out_total = out_ch * out_len;
    if (out_total > MAX_N)                     mp_raise_ValueError(MP_ERROR_TEXT("output would exceed 256 elements"));
    int32_t sig_buf[MAX_N], kern_buf[MAX_N], out_buf[MAX_N];
    for (size_t i = 0; i < sig_total;  i++) sig_buf[i]  = mp_obj_get_int(sig_items[i]);
    for (size_t i = 0; i < kern_total; i++) kern_buf[i] = mp_obj_get_int(kern_items[i]);
    conv1d(sig_buf, kern_buf, out_buf, in_ch, out_ch, k_len, sig_len);
    mp_obj_t result = mp_obj_new_list((size_t)out_total, NULL);
    mp_obj_list_t *result_list = MP_OBJ_TO_PTR(result);
    for (int32_t i = 0; i < out_total; i++) {
        result_list->items[i] = mp_obj_new_int(out_buf[i]);
    }
    return result;
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(fabric_conv1d_obj, 5, 5, fabric_conv1d);

// fabric.avg_pool1d(signal, window_size) -> non-overlapping window average
static mp_obj_t fabric_avg_pool1d(mp_obj_t signal_obj, mp_obj_t window_obj) {
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
    for (size_t i = 0; i < sig_len; i++) sig_buf[i] = mp_obj_get_int(items[i]);
    int32_t out_len = (int32_t)sig_len / w;
    avg_pool1d(sig_buf, out_buf, (int32_t)sig_len, w);
    mp_obj_t result = mp_obj_new_list((size_t)out_len, NULL);
    mp_obj_list_t *lst = MP_OBJ_TO_PTR(result);
    for (int32_t i = 0; i < out_len; i++) lst->items[i] = mp_obj_new_int(out_buf[i]);
    return result;
}
static MP_DEFINE_CONST_FUN_OBJ_2(fabric_avg_pool1d_obj, fabric_avg_pool1d);

// fabric.pointwise_conv(input, weights, in_ch, out_ch) -> list of int32
static mp_obj_t fabric_pointwise_conv(size_t n_args, const mp_obj_t *args) {
    (void)n_args;
    size_t inp_len, w_len;
    mp_obj_t *inp_items, *w_items;
    mp_obj_get_array(args[0], &inp_len, &inp_items);
    mp_obj_get_array(args[1], &w_len,   &w_items);
    int32_t in_ch  = mp_obj_get_int(args[2]);
    int32_t out_ch = mp_obj_get_int(args[3]);
    if (in_ch < 1 || in_ch > MAX_MATMUL)   mp_raise_ValueError(MP_ERROR_TEXT("in_ch must be 1..16"));
    if (out_ch < 1 || out_ch > MAX_MATMUL)  mp_raise_ValueError(MP_ERROR_TEXT("out_ch must be 1..16"));
    if ((int32_t)inp_len != in_ch)           mp_raise_ValueError(MP_ERROR_TEXT("input length must equal in_ch"));
    if ((int32_t)w_len != out_ch*in_ch)      mp_raise_ValueError(MP_ERROR_TEXT("weights length must equal out_ch*in_ch"));
    int8_t  inp_buf[MAX_MATMUL], w_buf[MAX_MATMUL*MAX_MATMUL];
    int32_t out_buf[MAX_MATMUL];
    for (int32_t i = 0; i < in_ch;        i++) inp_buf[i] = (int8_t)mp_obj_get_int(inp_items[i]);
    for (int32_t i = 0; i < out_ch*in_ch; i++) w_buf[i]   = (int8_t)mp_obj_get_int(w_items[i]);
    pointwise_conv(inp_buf, w_buf, out_buf, in_ch, out_ch);
    mp_obj_t result = mp_obj_new_list((size_t)out_ch, NULL);
    mp_obj_list_t *result_list = MP_OBJ_TO_PTR(result);
    for (int32_t i = 0; i < out_ch; i++) result_list->items[i] = mp_obj_new_int(out_buf[i]);
    return result;
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(fabric_pointwise_conv_obj, 4, 4, fabric_pointwise_conv);

// fabric.threshold(a, thresh) -> list: 1 if a[i] > thresh else 0
static mp_obj_t fabric_threshold(mp_obj_t a_obj, mp_obj_t thresh_obj) {
    size_t len;
    mp_obj_t *items;
    mp_obj_get_array(a_obj, &len, &items);
    if (len == 0 || len > MAX_N) mp_raise_ValueError(MP_ERROR_TEXT("list length must be 1..256"));
    int32_t thresh = mp_obj_get_int(thresh_obj);
    int32_t a_buf[MAX_N], out_buf[MAX_N];
    for (size_t i = 0; i < len; i++) a_buf[i] = mp_obj_get_int(items[i]);
    vec_threshold(a_buf, thresh, out_buf, (int32_t)len);
    mp_obj_t result = mp_obj_new_list(len, NULL);
    mp_obj_list_t *result_list = MP_OBJ_TO_PTR(result);
    for (size_t i = 0; i < len; i++) result_list->items[i] = mp_obj_new_int(out_buf[i]);
    return result;
}
static MP_DEFINE_CONST_FUN_OBJ_2(fabric_threshold_obj, fabric_threshold);

// fabric.vmax(a, b) -> list: element-wise max
static mp_obj_t fabric_vmax(mp_obj_t a_obj, mp_obj_t b_obj) {
    size_t a_len, b_len;
    mp_obj_t *a_items, *b_items;
    mp_obj_get_array(a_obj, &a_len, &a_items);
    mp_obj_get_array(b_obj, &b_len, &b_items);
    if (a_len != b_len) mp_raise_ValueError(MP_ERROR_TEXT("lists must be same length"));
    if (a_len == 0 || a_len > MAX_N) mp_raise_ValueError(MP_ERROR_TEXT("list length must be 1..256"));
    int32_t a_buf[MAX_N], b_buf[MAX_N], out_buf[MAX_N];
    for (size_t i = 0; i < a_len; i++) { a_buf[i] = mp_obj_get_int(a_items[i]); b_buf[i] = mp_obj_get_int(b_items[i]); }
    vec_vmax(a_buf, b_buf, out_buf, (int32_t)a_len);
    mp_obj_t result = mp_obj_new_list(a_len, NULL);
    mp_obj_list_t *result_list = MP_OBJ_TO_PTR(result);
    for (size_t i = 0; i < a_len; i++) result_list->items[i] = mp_obj_new_int(out_buf[i]);
    return result;
}
static MP_DEFINE_CONST_FUN_OBJ_2(fabric_vmax_obj, fabric_vmax);

// fabric.vmin(a, b) -> list: element-wise min
static mp_obj_t fabric_vmin(mp_obj_t a_obj, mp_obj_t b_obj) {
    size_t a_len, b_len;
    mp_obj_t *a_items, *b_items;
    mp_obj_get_array(a_obj, &a_len, &a_items);
    mp_obj_get_array(b_obj, &b_len, &b_items);
    if (a_len != b_len) mp_raise_ValueError(MP_ERROR_TEXT("lists must be same length"));
    if (a_len == 0 || a_len > MAX_N) mp_raise_ValueError(MP_ERROR_TEXT("list length must be 1..256"));
    int32_t a_buf[MAX_N], b_buf[MAX_N], out_buf[MAX_N];
    for (size_t i = 0; i < a_len; i++) { a_buf[i] = mp_obj_get_int(a_items[i]); b_buf[i] = mp_obj_get_int(b_items[i]); }
    vec_vmin(a_buf, b_buf, out_buf, (int32_t)a_len);
    mp_obj_t result = mp_obj_new_list(a_len, NULL);
    mp_obj_list_t *result_list = MP_OBJ_TO_PTR(result);
    for (size_t i = 0; i < a_len; i++) result_list->items[i] = mp_obj_new_int(out_buf[i]);
    return result;
}
static MP_DEFINE_CONST_FUN_OBJ_2(fabric_vmin_obj, fabric_vmin);

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

// fabric.multiply(a, b) → element-wise multiply (b list) or scalar multiply (b int)
static mp_obj_t fabric_multiply(mp_obj_t a_obj, mp_obj_t b_obj) {
    size_t a_len;
    mp_obj_t *a_items;
    mp_obj_get_array(a_obj, &a_len, &a_items);
    if (a_len == 0 || a_len > MAX_N) {
        mp_raise_ValueError(MP_ERROR_TEXT("list length must be 1..256"));
    }
    int32_t a_buf[MAX_N], out_buf[MAX_N];
    for (size_t i = 0; i < a_len; i++) {
        a_buf[i] = mp_obj_get_int(a_items[i]);
    }
    if (mp_obj_is_int(b_obj)) {
        // scalar broadcast
        int32_t scalar = mp_obj_get_int(b_obj);
        vec_scale(a_buf, scalar, out_buf, (int32_t)a_len);
    } else {
        // element-wise
        size_t b_len;
        mp_obj_t *b_items;
        mp_obj_get_array(b_obj, &b_len, &b_items);
        if (b_len != a_len) {
            mp_raise_ValueError(MP_ERROR_TEXT("lists must be same length"));
        }
        int32_t b_buf[MAX_N];
        for (size_t i = 0; i < b_len; i++) {
            b_buf[i] = mp_obj_get_int(b_items[i]);
        }
        mul(a_buf, b_buf, out_buf, (int32_t)a_len);
    }
    mp_obj_t result = mp_obj_new_list(a_len, NULL);
    mp_obj_list_t *result_list = MP_OBJ_TO_PTR(result);
    for (size_t i = 0; i < a_len; i++) {
        result_list->items[i] = mp_obj_new_int(out_buf[i]);
    }
    return result;
}
static MP_DEFINE_CONST_FUN_OBJ_2(fabric_multiply_obj, fabric_multiply);

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


// fabric.argmax(scores) → index of maximum value (no size limit — scans inline)
static mp_obj_t fabric_argmax(mp_obj_t a_obj) {
    size_t len;
    mp_obj_t *items;
    mp_obj_get_array(a_obj, &len, &items);
    if (len == 0) {
        mp_raise_ValueError(MP_ERROR_TEXT("list is empty"));
    }
    int32_t best_idx = 0;
    int32_t best_val = mp_obj_get_int(items[0]);
    for (size_t i = 1; i < len; i++) {
        int32_t v = mp_obj_get_int(items[i]);
        if (v > best_val) { best_val = v; best_idx = (int32_t)i; }
    }
    return mp_obj_new_int(best_idx);
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


// fabric.fft(signal) → list of 4096 [real, imag] pairs (Q15 int16)
// signal: list of 4096 int16 real values (imaginary part assumed 0)
// Uses the radix-4 FFT kernel from the E1x SDK example.
static mp_obj_t fabric_fft(mp_obj_t signal_obj) {
    size_t sig_len;
    mp_obj_t *sig_items;
    mp_obj_get_array(signal_obj, &sig_len, &sig_items);
    if (sig_len != FFT_SIZE) {
        mp_raise_ValueError(MP_ERROR_TEXT("signal must have 4096 elements"));
    }
    static fft_cpx src_buf[FFT_SIZE];
    static fft_cpx dst_buf[FFT_SIZE];
    for (size_t i = 0; i < FFT_SIZE; i++) {
        src_buf[i].r = (int16_t)mp_obj_get_int(sig_items[i]);
        src_buf[i].i = 0;
    }
    fft4(src_buf, dst_buf);
    mp_obj_t result = mp_obj_new_list(FFT_SIZE, NULL);
    mp_obj_list_t *result_list = MP_OBJ_TO_PTR(result);
    for (size_t i = 0; i < FFT_SIZE; i++) {
        mp_obj_t pair = mp_obj_new_list(2, NULL);
        mp_obj_list_t *pair_list = MP_OBJ_TO_PTR(pair);
        pair_list->items[0] = mp_obj_new_int(dst_buf[i].r);
        pair_list->items[1] = mp_obj_new_int(dst_buf[i].i);
        result_list->items[i] = pair;
    }
    return result;
}
static MP_DEFINE_CONST_FUN_OBJ_1(fabric_fft_obj, fabric_fft);

// fabric.fft_power(signal) → list of 4096 int32 power values (r²+i²)
// Useful for spectrum analysis — avoids boxing complex pairs.
static mp_obj_t fabric_fft_power(mp_obj_t signal_obj) {
    size_t sig_len;
    mp_obj_t *sig_items;
    mp_obj_get_array(signal_obj, &sig_len, &sig_items);
    if (sig_len != FFT_SIZE) {
        mp_raise_ValueError(MP_ERROR_TEXT("signal must have 4096 elements"));
    }
    static fft_cpx src_buf[FFT_SIZE];
    static fft_cpx dst_buf[FFT_SIZE];
    for (size_t i = 0; i < FFT_SIZE; i++) {
        src_buf[i].r = (int16_t)mp_obj_get_int(sig_items[i]);
        src_buf[i].i = 0;
    }
    fft4(src_buf, dst_buf);
    mp_obj_t result = mp_obj_new_list(FFT_SIZE, NULL);
    mp_obj_list_t *result_list = MP_OBJ_TO_PTR(result);
    for (size_t i = 0; i < FFT_SIZE; i++) {
        int32_t r = dst_buf[i].r, im = dst_buf[i].i;
        result_list->items[i] = mp_obj_new_int(r*r + im*im);
    }
    return result;
}
static MP_DEFINE_CONST_FUN_OBJ_1(fabric_fft_power_obj, fabric_fft_power);

// fabric.softmax(scores, scale=256) → list of probabilities × scale (int)
// scores: list of int values (e.g. int8 logits from classification)
// scale: output scale factor (default 256 gives 8-bit probabilities)
// Uses Q15 fixed-point exp via lookup table; runs on scalar core.
static const int32_t _exp_lut[256] = {
    // exp(-k/8) * 32768 for k = 0..255, clamped to int32
    // k=0: exp(0)=1.0 → 32768
    32768,30338,28087,26001,24074,22295,20655,19144,17755,16482,
    15262,14161,13128,12160,11252,10408, 9631, 8917, 8264, 7670,
     7132, 6644, 6202, 5804, 5444, 5120, 4826, 4558, 4313, 4087,
     3879, 3685, 3504, 3334, 3175, 3024, 2881, 2746, 2618, 2497,
     2382, 2272, 2168, 2069, 1974, 1885, 1799, 1717, 1639, 1565,
     1494, 1427, 1362, 1300, 1241, 1185, 1131, 1080, 1031,  984,
      939,  897,  856,  816,  779,  744,  710,  677,  646,  617,
      588,  562,  536,  512,  488,  466,  445,  424,  405,  387,
      369,  352,  336,  320,  306,  292,  278,  266,  254,  242,
      231,  220,  210,  201,  191,  183,  174,  166,  158,  151,
      144,  138,  131,  125,  120,  114,  109,  104,   99,   94,
       90,   86,   82,   78,   75,   71,   68,   65,   62,   59,
       56,   54,   51,   49,   47,   44,   42,   40,   38,   37,
       35,   33,   32,   30,   29,   27,   26,   25,   24,   22,
       21,   20,   19,   18,   18,   17,   16,   15,   14,   14,
       13,   13,   12,   11,   11,   10,   10,    9,    9,    9,
        8,    8,    7,    7,    7,    6,    6,    6,    5,    5,
        5,    5,    4,    4,    4,    4,    4,    3,    3,    3,
        3,    3,    3,    2,    2,    2,    2,    2,    2,    2,
        2,    2,    1,    1,    1,    1,    1,    1,    1,    1,
        1,    1,    1,    1,    1,    1,    1,    1,    0,    0,
        0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
        0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
        0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
        0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
        0,    0,    0,    0,    0,    0
};

static mp_obj_t fabric_softmax(size_t n_args, const mp_obj_t *args) {
    (void)n_args;
    size_t len;
    mp_obj_t *items;
    mp_obj_get_array(args[0], &len, &items);
    if (len == 0 || len > MAX_N) {
        mp_raise_ValueError(MP_ERROR_TEXT("list length must be 1..256"));
    }
    int32_t scale = (n_args > 1) ? mp_obj_get_int(args[1]) : 256;

    // find max for numerical stability
    int32_t max_val = mp_obj_get_int(items[0]);
    for (size_t i = 1; i < len; i++) {
        int32_t v = mp_obj_get_int(items[i]);
        if (v > max_val) max_val = v;
    }

    // compute exp(x - max) using LUT: index = clamp((max - x) * 8, 0, 255)
    int32_t exp_vals[MAX_N];
    int32_t sum = 0;
    for (size_t i = 0; i < len; i++) {
        int32_t diff = (max_val - mp_obj_get_int(items[i])) * 8;
        int32_t idx = diff < 0 ? 0 : diff > 255 ? 255 : diff;
        exp_vals[i] = _exp_lut[idx];
        sum += exp_vals[i];
    }

    // normalise: out[i] = exp_vals[i] * scale / sum
    mp_obj_t result = mp_obj_new_list(len, NULL);
    mp_obj_list_t *result_list = MP_OBJ_TO_PTR(result);
    for (size_t i = 0; i < len; i++) {
        int32_t prob = (sum > 0) ? (exp_vals[i] * scale) / sum : 0;
        result_list->items[i] = mp_obj_new_int(prob);
    }
    return result;
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(fabric_softmax_obj, 1, 2, fabric_softmax);

// fabric.ticks_us() → microseconds since boot (wraps at 2^32)
static mp_obj_t fabric_ticks_us(void) {
    return mp_obj_new_int_from_uint((mp_uint_t)eff_mtimer_uptime_us());
}
static MP_DEFINE_CONST_FUN_OBJ_0(fabric_ticks_us_obj, fabric_ticks_us);

static const mp_rom_map_elem_t fabric_module_globals_table[] = {
    { MP_ROM_QSTR(MP_QSTR___name__), MP_ROM_QSTR(MP_QSTR_fabric) },
    { MP_ROM_QSTR(MP_QSTR_ticks_us), MP_ROM_PTR(&fabric_ticks_us_obj) },
    { MP_ROM_QSTR(MP_QSTR_fft), MP_ROM_PTR(&fabric_fft_obj) },
    { MP_ROM_QSTR(MP_QSTR_fft_power), MP_ROM_PTR(&fabric_fft_power_obj) },
    { MP_ROM_QSTR(MP_QSTR_softmax), MP_ROM_PTR(&fabric_softmax_obj) },
    { MP_ROM_QSTR(MP_QSTR_matmul_int8_rq), MP_ROM_PTR(&fabric_matmul_int8_rq_obj) },
    { MP_ROM_QSTR(MP_QSTR_conv2d_int8_rq), MP_ROM_PTR(&fabric_conv2d_int8_rq_obj) },
    { MP_ROM_QSTR(MP_QSTR_pointwise_conv_rq), MP_ROM_PTR(&fabric_pointwise_conv_rq_obj) },
    { MP_ROM_QSTR(MP_QSTR_requantize), MP_ROM_PTR(&fabric_requantize_obj) },
    { MP_ROM_QSTR(MP_QSTR_sum_squares), MP_ROM_PTR(&fabric_sum_squares_obj) },
    { MP_ROM_QSTR(MP_QSTR_l2_norm), MP_ROM_PTR(&fabric_l2_norm_obj) },
    { MP_ROM_QSTR(MP_QSTR_biquad_cascade), MP_ROM_PTR(&fabric_biquad_cascade_obj) },
    { MP_ROM_QSTR(MP_QSTR_conv2d_int8), MP_ROM_PTR(&fabric_conv2d_int8_obj) },
    { MP_ROM_QSTR(MP_QSTR_conv1d), MP_ROM_PTR(&fabric_conv1d_obj) },
    { MP_ROM_QSTR(MP_QSTR_avg_pool1d), MP_ROM_PTR(&fabric_avg_pool1d_obj) },
    { MP_ROM_QSTR(MP_QSTR_pointwise_conv), MP_ROM_PTR(&fabric_pointwise_conv_obj) },
    { MP_ROM_QSTR(MP_QSTR_threshold), MP_ROM_PTR(&fabric_threshold_obj) },
    { MP_ROM_QSTR(MP_QSTR_maximum), MP_ROM_PTR(&fabric_vmax_obj) },
    { MP_ROM_QSTR(MP_QSTR_minimum), MP_ROM_PTR(&fabric_vmin_obj) },
    { MP_ROM_QSTR(MP_QSTR_relu), MP_ROM_PTR(&fabric_relu_obj) },
    { MP_ROM_QSTR(MP_QSTR_multiply), MP_ROM_PTR(&fabric_multiply_obj) },
    { MP_ROM_QSTR(MP_QSTR_add), MP_ROM_PTR(&fabric_add_obj) },
    { MP_ROM_QSTR(MP_QSTR_max_pool1d), MP_ROM_PTR(&fabric_max_pool1d_obj) },
    { MP_ROM_QSTR(MP_QSTR_matmul), MP_ROM_PTR(&fabric_matmul_obj) },
    { MP_ROM_QSTR(MP_QSTR_clip), MP_ROM_PTR(&fabric_clip_obj) },
    { MP_ROM_QSTR(MP_QSTR_biquad), MP_ROM_PTR(&fabric_biquad_obj) },
    { MP_ROM_QSTR(MP_QSTR_matmul_int8), MP_ROM_PTR(&fabric_matmul_int8_obj) },
    { MP_ROM_QSTR(MP_QSTR_dot), MP_ROM_PTR(&fabric_dot_product_obj) },
    { MP_ROM_QSTR(MP_QSTR_matvec), MP_ROM_PTR(&fabric_matvec_obj) },
    { MP_ROM_QSTR(MP_QSTR_fir), MP_ROM_PTR(&fabric_fir_obj) },
    { MP_ROM_QSTR(MP_QSTR_argmax), MP_ROM_PTR(&fabric_argmax_obj) },
};
static MP_DEFINE_CONST_DICT(fabric_module_globals, fabric_module_globals_table);

const mp_obj_module_t fabric_module = {
    .base = { &mp_type_module },
    .globals = (mp_obj_dict_t *)&fabric_module_globals,
};

MP_REGISTER_MODULE(MP_QSTR_fabric, fabric_module);
