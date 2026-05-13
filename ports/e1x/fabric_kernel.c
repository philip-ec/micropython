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
