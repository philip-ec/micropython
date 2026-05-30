#include <stdint.h>
#include "py/obj.h"
#include "py/runtime.h"

#define MAX_N        256
#define MAX_ROWS      32
#define MAX_COLS      32
#define MAX_MATMUL    16

int32_t dot_product(const int32_t *a, const int32_t *b, int32_t n);
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
