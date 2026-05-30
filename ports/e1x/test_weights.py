import weights
import fabric

# Demo MLP: 4 -> 4 -> 2
# weights frozen in firmware via weights_gen.py + demo_weights.json

x = [10, 20, 30, 40]

# fc1: 4x4 matmul + requantize (scale=1, shift=7, zero_point=0)
h = weights.fc1(x, 1, 7, 0)
print("fc1:", h)
assert h == [-18, 12, 17, 17]

# fc2: 2x4 matmul + requantize
out = weights.fc2(h, 1, 7, 0)
print("fc2:", out)
assert out == [-15, -20]

print("weights OK")
print("argmax:", fabric.argmax(out))
