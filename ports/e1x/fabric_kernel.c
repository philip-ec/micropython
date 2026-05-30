#include <stdint.h>
#include <eff.h>

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
