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
