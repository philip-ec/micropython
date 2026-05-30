import weights, fabric, testdata

correct = 0
total_us = 0
for d in range(10):
    x = testdata.digit(d)
    t0 = fabric.ticks_us()
    h = weights.fc1(x, 1, 12, 0)
    h = fabric.relu(h)
    out = weights.fc2(h, 1, 11, 0)
    t1 = fabric.ticks_us()
    pred = fabric.argmax(out)
    us = t1 - t0
    total_us += us
    ok = pred == d
    if ok:
        correct += 1
    print(d, ">", pred, us, "us", "OK" if ok else "FAIL")
print("accuracy:", correct, "/ 10")
print("avg:", total_us // 10, "us per inference")
