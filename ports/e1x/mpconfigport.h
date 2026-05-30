#include <stdint.h>

#define MICROPY_CONFIG_ROM_LEVEL        (MICROPY_CONFIG_ROM_LEVEL_MINIMUM)
#define MICROPY_LONGINT_IMPL            (MICROPY_LONGINT_IMPL_MPZ)

#define MICROPY_ENABLE_COMPILER         (1)
#define MICROPY_ENABLE_GC               (1)
#define MICROPY_HELPER_REPL             (1)
#define MICROPY_PY_BUILTINS_MIN_MAX     (1)
#define MICROPY_PY_MATH                 (1)
#define MICROPY_ENABLE_EXTERNAL_IMPORT  (0)
#define MICROPY_ALLOC_PATH_MAX          (64)
#define MICROPY_ALLOC_PARSE_CHUNK_INIT  (16)

#define MICROPY_PY_SYS_MODULES          (0)
#define MICROPY_PY_SYS_EXIT             (0)
#define MICROPY_PY_SYS_PATH             (0)
#define MICROPY_PY_SYS_ARGV             (0)

typedef long mp_off_t;

// system alloca.h uses glibc macros not present in E1x libc
#define alloca __builtin_alloca

#define MICROPY_HEAP_SIZE               (2 * 1024 * 1024)

#define MICROPY_HW_BOARD_NAME           "e1x-evk"
#define MICROPY_HW_MCU_NAME             "efficient-e1x"

#define MP_STATE_PORT                   MP_STATE_VM

