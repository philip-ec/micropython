import fabric

# softmax on MNIST scores
scores = [-1, -5, 1, 4, -4, 0, -12, 12, 0, 1]
probs = fabric.softmax(scores, 1000)
print("softmax:", probs)
print("sum:", sum(probs))
assert fabric.argmax(probs) == 7
print("softmax OK")

# FFT: DC signal (all 1000) -> all energy at DC bin 0
# Output is bit-reversed: DC bin 0 stays at position 0
sig = [1000] * 4096
t0 = fabric.ticks_us()
power = fabric.fft_power(sig)
t1 = fabric.ticks_us()
print("FFT len:", len(power))
print("FFT time:", t1 - t0, "us")
print("power[0]:", power[0], "power[1]:", power[1], "power[2]:", power[2])
peak = fabric.argmax(power)
print("FFT peak bin:", peak, " (expected 0 for DC)")
assert peak == 0
print("fft OK")
