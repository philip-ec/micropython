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
