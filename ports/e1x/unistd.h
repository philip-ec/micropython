// Minimal unistd.h stub for the E1x bare-metal port.
// MicroPython's py/ core needs ssize_t and SEEK_* from this header.
#pragma once
#include <llvm-libc-types/ssize_t.h>
#include <llvm-libc-macros/file-seek-macros.h>
