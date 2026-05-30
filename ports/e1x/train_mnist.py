#!/usr/bin/env python3
"""
Train a 784->128->10 int8-quantized MLP on MNIST and export weights.json.

Requirements: numpy + tensorflow (for MNIST data only).

Usage:
    python3 train_mnist.py
    # produces mnist_weights.json, then:
    python3 weights_gen.py mnist_weights.json
    make && eff-flash build/firmware.hex sram -p /dev/ttyACM0
"""
import numpy as np
import json

# ── Data loading ─────────────────────────────────────────────────────────────

def load_mnist():
    print('Loading MNIST via Keras...')
    import tensorflow as tf
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.reshape(-1, 784).astype(np.float32)
    X_test  = X_test.reshape(-1, 784).astype(np.float32)
    return X_train, y_train.astype(np.int32), X_test, y_test.astype(np.int32)

# ── MLP helpers ───────────────────────────────────────────────────────────────

def relu(x): return np.maximum(0, x)

def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def cross_entropy(probs, labels):
    return -np.log(probs[np.arange(len(labels)), labels] + 1e-9).mean()

# ── Training ──────────────────────────────────────────────────────────────────

def train(X_train, y_train, X_test, y_test,
          hidden=128, lr=0.05, epochs=30, batch=256, seed=42):
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((784, hidden)).astype(np.float32) * np.sqrt(2/784)
    b1 = np.zeros(hidden, dtype=np.float32)
    W2 = rng.standard_normal((hidden, 10)).astype(np.float32) * np.sqrt(2/hidden)
    b2 = np.zeros(10, dtype=np.float32)

    for epoch in range(epochs):
        idx = rng.permutation(len(X_train))
        total_loss = 0.0
        for start in range(0, len(X_train), batch):
            xb = X_train[idx[start:start+batch]]
            yb = y_train[idx[start:start+batch]]
            z1 = xb @ W1 + b1; a1 = relu(z1)
            z2 = a1 @ W2 + b2; probs = softmax(z2)
            loss = cross_entropy(probs, yb); total_loss += loss
            nb = len(xb)
            dz2 = probs.copy(); dz2[np.arange(nb), yb] -= 1; dz2 /= nb
            dW2 = a1.T @ dz2; db2 = dz2.sum(0)
            da1 = dz2 @ W2.T; dz1 = da1 * (z1 > 0)
            dW1 = xb.T @ dz1; db1 = dz1.sum(0)
            W1 -= lr*dW1; b1 -= lr*db1; W2 -= lr*dW2; b2 -= lr*db2

        z1 = X_test @ W1 + b1; a1 = relu(z1)
        acc = ((a1 @ W2 + b2).argmax(1) == y_test).mean()
        print(f'epoch {epoch+1:2d}/{epochs}  loss={total_loss/(len(X_train)//batch):.4f}  test_acc={acc:.4f}')

    return W1, b1, W2, b2

# ── Quantization ──────────────────────────────────────────────────────────────

def quantize_weights(W):
    """Symmetric int8, returns (W_int8, scale) where W_float ≈ W_int8 * scale."""
    scale = float(np.abs(W).max()) / 127.0
    W_q = np.clip(np.round(W / scale), -128, 127).astype(np.int8)
    return W_q, scale

def compute_rq_params(w_scale, act_max):
    """
    Compute (scale_int, shift) for requantize so that:
      out_int8 ≈ int32_acc * w_scale / (127 * act_max) * 127
               = int32_acc * w_scale / act_max

    We need: scale_int / 2^shift ≈ w_scale / act_max
    """
    multiplier = w_scale / act_max
    # Find shift that gives a scale_int in [1, 2^30]
    for shift in range(1, 31):
        scale_int = int(round(multiplier * (1 << shift)))
        if scale_int >= 1:
            return scale_int, shift
    return 1, 0

def simulate_int8(X_int8, W1_q, b1_q, W2_q, b2_q, s1, sh1, s2, sh2):
    """Simulate int8 inference to verify accuracy."""
    correct = 0
    for i in range(len(X_int8)):
        x = X_int8[i].astype(np.int32)
        # fc1 + bias + requantize + relu
        acc1 = W1_q.T.astype(np.int32) @ x + b1_q
        h = np.clip((acc1 * s1) >> sh1, -128, 127).astype(np.int8)
        h = np.maximum(h, 0)
        # fc2 + bias + requantize
        acc2 = W2_q.T.astype(np.int32) @ h.astype(np.int32) + b2_q
        out = np.clip((acc2 * s2) >> sh2, -128, 127).astype(np.int8)
        if int(out.argmax()) == int(y_test[i]):
            correct += 1
    return correct / len(X_int8)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global y_test
    print('Loading MNIST...')
    X_train, y_train, X_test, y_test = load_mnist()

    # Normalize pixels to [-1, 1] float, then to int8
    X_train_f = (X_train - 128.0) / 128.0
    X_test_f  = (X_test  - 128.0) / 128.0
    X_test_i8 = np.clip(np.round(X_test_f * 127), -128, 127).astype(np.int8)

    print(f'Train: {X_train_f.shape}, Test: {X_test_f.shape}')
    print('Training...')
    W1, b1, W2, b2 = train(X_train_f, y_train, X_test_f, y_test,
                            hidden=128, lr=0.05, epochs=30, batch=256)

    z1_f = X_test_f @ W1 + b1
    float_acc = ((relu(z1_f) @ W2 + b2).argmax(1) == y_test).mean()
    print(f'\nFloat accuracy: {float_acc:.4f}')

    # Quantize weights
    W1_q, w1_scale = quantize_weights(W1)
    W2_q, w2_scale = quantize_weights(W2)

    # Quantize biases in accumulator units:
    # acc = W_q.T @ x_int8, float equiv = acc * w_scale / 127
    # bias_float = b  →  bias_q = round(b * 127 / w_scale)
    b1_q = np.clip(np.round(b1 * 127 / w1_scale), -(2**30), 2**30-1).astype(np.int32)
    b2_q = np.clip(np.round(b2 * 127 / w2_scale), -(2**30), 2**30-1).astype(np.int32)

    # Calibrate requantize using actual activation ranges on test set
    z1 = X_test_f @ W1 + b1; a1 = relu(z1)
    a1_max = float(np.abs(a1).max())
    z2 = a1 @ W2 + b2
    a2_max = float(np.abs(z2).max())

    print(f'\nActivation ranges: a1_max={a1_max:.4f}, a2_max={a2_max:.4f}')

    s1, sh1 = compute_rq_params(w1_scale, a1_max)
    s2, sh2 = compute_rq_params(w2_scale, a2_max)

    print(f'fc1: w_scale={w1_scale:.6f}, rq scale={s1}, shift={sh1}')
    print(f'fc2: w_scale={w2_scale:.6f}, rq scale={s2}, shift={sh2}')

    # Verify int8 accuracy
    print('\nSimulating int8 inference...')
    int8_acc = simulate_int8(X_test_i8, W1_q, b1_q, W2_q, b2_q, s1, sh1, s2, sh2)
    print(f'Int8 accuracy: {int8_acc:.4f}')

    if int8_acc < 0.90:
        print('WARNING: int8 accuracy below 90% — quantization may need tuning.')

    # Export JSON — weights transposed to (rows=out, cols=in) for C matmul
    spec = {
        'fc1': {
            'rows': W1_q.shape[1],   # 128
            'cols': W1_q.shape[0],   # 784
            'weights': W1_q.T.tolist(),
            'bias':    b1_q.tolist(),
            'scale': s1, 'shift': sh1, 'zero_point': 0,
        },
        'fc2': {
            'rows': W2_q.shape[1],   # 10
            'cols': W2_q.shape[0],   # 128
            'weights': W2_q.T.tolist(),
            'bias':    b2_q.tolist(),
            'scale': s2, 'shift': sh2, 'zero_point': 0,
        },
    }

    out_path = 'mnist_weights.json'
    with open(out_path, 'w') as f:
        json.dump(spec, f)
    print(f'\nExported {out_path}')
    print('Next:')
    print('  python3 weights_gen.py mnist_weights.json')
    print('  make && eff-flash build/firmware.hex sram -p /dev/ttyACM0')

    # Print a sample for REPL testing
    sample = X_test_i8[0].tolist()
    print(f'\nTest digit 0: label={y_test[0]}')
    print(f'x = {sample}')

if __name__ == '__main__':
    main()
