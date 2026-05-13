#include <stdio.h>
#include <eff.h>
#include <eff/drivers/uart.h>
#include "py/mpconfig.h"

int mp_hal_stdin_rx_chr(void) {
    char c;
    while (eff_uart_rx_empty(STDIO_UART)) {}
    eff_uart_getc(STDIO_UART, &c);
    return (unsigned char)c;
}

mp_uint_t mp_hal_stdout_tx_strn(const char *str, mp_uint_t len) {
    fwrite(str, 1, len, stdout);
    fflush(stdout);
    return len;
}
