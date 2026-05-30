#include <stdint.h>
#include <eff.h>

__efficient__
void matmul_int8_rq(const int8_t *a, const int8_t *b, int8_t *out,
                    int32_t M, int32_t K, int32_t N,
                    int32_t scale, int32_t shift, int32_t zero_point) {
    for (int32_t i = 0; i < M; i++) {
        for (int32_t j = 0; j < N; j++) {
            int32_t acc = 0;
            for (int32_t k = 0; k < K; k++) {
                acc += (int32_t)a[i*K+k] * (int32_t)b[k*N+j];
            }
            int32_t v = ((acc * scale) >> shift) + zero_point;
            if (v < -128) v = -128;
            if (v >  127) v =  127;
            out[i*N+j] = (int8_t)v;
        }
    }
}

__efficient__
void conv2d_int8_rq(const int8_t *input, const int8_t *kernel, int8_t *out,
                    int32_t in_h, int32_t in_w, int32_t k_h, int32_t k_w,
                    int32_t scale, int32_t shift, int32_t zero_point) {
    int32_t out_h = in_h - k_h + 1;
    int32_t out_w = in_w - k_w + 1;
    for (int32_t oy = 0; oy < out_h; oy++) {
        for (int32_t ox = 0; ox < out_w; ox++) {
            int32_t acc = 0;
            for (int32_t ky = 0; ky < k_h; ky++) {
                for (int32_t kx = 0; kx < k_w; kx++) {
                    acc += (int32_t)input[(oy+ky)*in_w+(ox+kx)]
                         * (int32_t)kernel[ky*k_w+kx];
                }
            }
            int32_t v = ((acc * scale) >> shift) + zero_point;
            if (v < -128) v = -128;
            if (v >  127) v =  127;
            out[oy*out_w+ox] = (int8_t)v;
        }
    }
}

__efficient__
void pointwise_conv_rq(const int8_t *input, const int8_t *weights, int8_t *out,
                       int32_t in_ch, int32_t out_ch,
                       int32_t scale, int32_t shift, int32_t zero_point) {
    for (int32_t i = 0; i < out_ch; i++) {
        int32_t acc = 0;
        for (int32_t j = 0; j < in_ch; j++) {
            acc += (int32_t)weights[i*in_ch+j] * (int32_t)input[j];
        }
        int32_t v = ((acc * scale) >> shift) + zero_point;
        if (v < -128) v = -128;
        if (v >  127) v =  127;
        out[i] = (int8_t)v;
    }
}

__efficient__
int32_t dot_product(const int32_t *a, const int32_t *b, int32_t n) {
    int32_t sum = 0;
    for (int32_t i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

__efficient__
void matvec(const int32_t *mat, const int32_t *vec, int32_t *out,
            int32_t rows, int32_t cols) {
    for (int32_t i = 0; i < rows; i++) {
        int32_t sum = 0;
        for (int32_t j = 0; j < cols; j++) {
            sum += mat[i * cols + j] * vec[j];
        }
        out[i] = sum;
    }
}

// Direct Form I biquad IIR. Coefficients in Q15 (scale floats by 32768).
// a1, a2 are the feedback coefficients (sign matches standard notation:
// y[n] = (b0*x[n] + b1*x1 + b2*x2 - a1*y1 - a2*y2) >> 15
__efficient__
void biquad(const int32_t *x, int32_t *y, int32_t n,
            int32_t b0, int32_t b1, int32_t b2,
            int32_t a1, int32_t a2) {
    int32_t x1 = 0, x2 = 0, y1 = 0, y2 = 0;
    for (int32_t i = 0; i < n; i++) {
        int32_t out = (b0 * x[i] + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2) >> 15;
        x2 = x1; x1 = x[i];
        y2 = y1; y1 = out;
        y[i] = out;
    }
}

__efficient__
void requantize(const int32_t *acc, int8_t *out, int32_t n,
                int32_t scale, int32_t shift, int32_t zero_point) {
    for (int32_t i = 0; i < n; i++) {
        int32_t v = ((acc[i] * scale) >> shift) + zero_point;
        if (v < -128) v = -128;
        if (v > 127)  v = 127;
        out[i] = (int8_t)v;
    }
}

__efficient__
int32_t sum_squares(const int32_t *a, int32_t n) {
    int32_t acc = 0;
    for (int32_t i = 0; i < n; i++) {
        acc += a[i] * a[i];
    }
    return acc;
}

// isqrt_i32: scalar helper (Newton's method), called by l2_norm on scalar side
static int32_t isqrt_i32(int32_t n) {
    if (n <= 0) return 0;
    int32_t x = n, y = (x + 1) >> 1;
    while (y < x) { x = y; y = (x + n / x) >> 1; }
    return x;
}

// l2_norm runs on scalar (isqrt not Fabric-eligible); sum_squares is Fabric-accelerated
void l2_norm(const int32_t *a, int32_t *out, int32_t n, int32_t scale) {
    int32_t s = isqrt_i32(sum_squares(a, n));
    if (s == 0) { for (int32_t i = 0; i < n; i++) out[i] = 0; return; }
    for (int32_t i = 0; i < n; i++) {
        out[i] = (a[i] * scale) / s;
    }
}

void biquad_cascade(const int32_t *signal, const int32_t *coeffs,
                    int32_t *out, int32_t *buf,
                    int32_t n, int32_t n_sections) {
    for (int32_t i = 0; i < n; i++) out[i] = signal[i];
    for (int32_t s = 0; s < n_sections; s++) {
        int32_t b0 = coeffs[s*5+0], b1 = coeffs[s*5+1], b2 = coeffs[s*5+2];
        int32_t a1 = coeffs[s*5+3], a2 = coeffs[s*5+4];
        int32_t x1 = 0, x2 = 0, y1 = 0, y2 = 0;
        for (int32_t i = 0; i < n; i++) {
            int32_t xi = out[i];
            int32_t yi = (b0*xi + b1*x1 + b2*x2 - a1*y1 - a2*y2) >> 15;
            x2 = x1; x1 = xi; y2 = y1; y1 = yi;
            buf[i] = yi;
        }
        for (int32_t i = 0; i < n; i++) out[i] = buf[i];
    }
}

__efficient__
void conv2d_int8(const int8_t *input, const int8_t *kernel, int32_t *out,
                 int32_t in_h, int32_t in_w, int32_t k_h, int32_t k_w) {
    int32_t out_h = in_h - k_h + 1;
    int32_t out_w = in_w - k_w + 1;
    for (int32_t oy = 0; oy < out_h; oy++) {
        for (int32_t ox = 0; ox < out_w; ox++) {
            int32_t acc = 0;
            for (int32_t ky = 0; ky < k_h; ky++) {
                for (int32_t kx = 0; kx < k_w; kx++) {
                    acc += (int32_t)input[(oy+ky)*in_w+(ox+kx)]
                         * (int32_t)kernel[ky*k_w+kx];
                }
            }
            out[oy*out_w+ox] = acc;
        }
    }
}

__efficient__
void conv1d(const int32_t *signal, const int32_t *kernels, int32_t *out,
            int32_t in_ch, int32_t out_ch, int32_t k_len, int32_t sig_len) {
    int32_t out_len = sig_len - k_len + 1;
    for (int32_t i = 0; i < out_len; i++) {
        for (int32_t oc = 0; oc < out_ch; oc++) {
            int32_t acc = 0;
            for (int32_t ic = 0; ic < in_ch; ic++) {
                int32_t base_k = oc * in_ch * k_len + ic * k_len;
                for (int32_t k = 0; k < k_len; k++) {
                    acc += signal[(i+k)*in_ch+ic] * kernels[base_k+k];
                }
            }
            out[i*out_ch+oc] = acc;
        }
    }
}

__efficient__
void avg_pool1d(const int32_t *signal, int32_t *out, int32_t n, int32_t w) {
    int32_t out_len = n / w;
    for (int32_t i = 0; i < out_len; i++) {
        int32_t base = i * w, sum = 0;
        for (int32_t j = 0; j < w; j++) sum += signal[base+j];
        out[i] = sum / w;
    }
}

__efficient__
void pointwise_conv(const int8_t *input, const int8_t *weights, int32_t *out,
                    int32_t in_ch, int32_t out_ch) {
    for (int32_t i = 0; i < out_ch; i++) {
        int32_t sum = 0;
        for (int32_t j = 0; j < in_ch; j++) {
            sum += (int32_t)weights[i*in_ch+j] * (int32_t)input[j];
        }
        out[i] = sum;
    }
}

__efficient__
void vec_threshold(const int32_t *a, int32_t thresh, int32_t *out, int32_t n) {
    for (int32_t i = 0; i < n; i++) out[i] = (a[i] > thresh) ? 1 : 0;
}

__efficient__
void vec_vmax(const int32_t *a, const int32_t *b, int32_t *out, int32_t n) {
    for (int32_t i = 0; i < n; i++) out[i] = (a[i] > b[i]) ? a[i] : b[i];
}

__efficient__
void vec_vmin(const int32_t *a, const int32_t *b, int32_t *out, int32_t n) {
    for (int32_t i = 0; i < n; i++) out[i] = (a[i] < b[i]) ? a[i] : b[i];
}

__efficient__
void relu(const int32_t *x, int32_t *out, int32_t n) {
    for (int32_t i = 0; i < n; i++) {
        out[i] = x[i] > 0 ? x[i] : 0;
    }
}

__efficient__
void vec_scale(const int32_t *a, int32_t scalar, int32_t *out, int32_t n) {
    for (int32_t i = 0; i < n; i++) {
        out[i] = a[i] * scalar;
    }
}

__efficient__
void vec_add(const int32_t *a, const int32_t *b, int32_t *out, int32_t n) {
    for (int32_t i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

__efficient__
void max_pool1d(const int32_t *signal, int32_t *out, int32_t n, int32_t w) {
    int32_t out_len = n / w;
    for (int32_t i = 0; i < out_len; i++) {
        int32_t base = i * w;
        int32_t mx = signal[base];
        for (int32_t j = 1; j < w; j++) {
            int32_t v = signal[base + j];
            if (v > mx) mx = v;
        }
        out[i] = mx;
    }
}

__efficient__
void matmul(const int32_t *a, const int32_t *b, int32_t *out,
            int32_t M, int32_t K, int32_t N) {
    for (int32_t i = 0; i < M; i++) {
        for (int32_t j = 0; j < N; j++) {
            int32_t sum = 0;
            for (int32_t k = 0; k < K; k++) {
                sum += a[i * K + k] * b[k * N + j];
            }
            out[i * N + j] = sum;
        }
    }
}

__efficient__
void clip(const int32_t *a, int32_t *out, int32_t n, int32_t lo, int32_t hi) {
    for (int32_t i = 0; i < n; i++) {
        int32_t v = a[i];
        if (v < lo) v = lo;
        if (v > hi) v = hi;
        out[i] = v;
    }
}

__efficient__
void matmul_int8(const int8_t *a, const int8_t *b, int32_t *out,
                 int32_t M, int32_t K, int32_t N) {
    for (int32_t i = 0; i < M; i++) {
        for (int32_t j = 0; j < N; j++) {
            int32_t sum = 0;
            for (int32_t k = 0; k < K; k++) {
                sum += (int32_t)a[i * K + k] * (int32_t)b[k * N + j];
            }
            out[i * N + j] = sum;
        }
    }
}

__efficient__
void mul(const int32_t *a, const int32_t *b, int32_t *out, int32_t n) {
    for (int32_t i = 0; i < n; i++) {
        out[i] = a[i] * b[i];
    }
}

__efficient__
int32_t argmax(const int32_t *a, int32_t n) {
    int32_t best_idx = 0;
    int32_t best_val = a[0];
    for (int32_t i = 1; i < n; i++) {
        if (a[i] > best_val) {
            best_val = a[i];
            best_idx = i;
        }
    }
    return best_idx;
}

__efficient__
void fir(const int32_t *signal, const int32_t *coeffs, int32_t *out,
         int32_t sig_len, int32_t n_coeffs) {
    int32_t out_len = sig_len - n_coeffs + 1;
    for (int32_t i = 0; i < out_len; i++) {
        int32_t sum = 0;
        for (int32_t j = 0; j < n_coeffs; j++) {
            sum += signal[i + j] * coeffs[j];
        }
        out[i] = sum;
    }
}
