#!/usr/bin/env python3
"""Send a .py file to MicroPython REPL via paste mode (Ctrl+E / Ctrl+D)."""
import sys
import time
import serial

PORT = "/dev/ttyACM2"
BAUD = 115200

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <file.py>")
    sys.exit(1)

with open(sys.argv[1]) as f:
    code = f.read()

with serial.Serial(PORT, BAUD, timeout=2) as ser:
    # interrupt any running code
    ser.write(b"\x03\x03")
    time.sleep(0.2)
    ser.read_all()

    # enter paste mode
    ser.write(b"\x05")
    time.sleep(0.1)

    # send file content
    for line in code.splitlines():
        ser.write((line + "\n").encode())
        time.sleep(0.1)

    # execute
    ser.write(b"\x04")
    time.sleep(0.5)

    # read and print output — wait up to 30s, reset on each new chunk
    deadline = time.time() + 30
    while time.time() < deadline:
        data = ser.read_all()
        if data:
            print(data.decode(errors="replace"), end="", flush=True)
            deadline = time.time() + 5
        else:
            time.sleep(0.1)
