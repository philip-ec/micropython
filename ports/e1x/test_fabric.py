import fabric

# dot_product
assert fabric.dot_product([1,2,3,4],[4,3,2,1]) == 20
assert fabric.dot_product([0]*10,[1]*10) == 0
assert fabric.dot_product([1]*256,[1]*256) == 256
print("dot_product OK")

# matvec
assert fabric.matvec([[1,0],[0,1]], [3,4]) == [3,4]
assert fabric.matvec([[1,2],[3,4]], [1,1]) == [3,7]
print("matvec OK")

# fir
assert fabric.fir([1,2,3,4,5],[1,1,1]) == [6,9,12]
assert fabric.fir([10,20,30],[1]) == [10,20,30]
assert fabric.fir([1,3,6,10],[-1,1]) == [2,3,4]
print("fir OK")

# argmax
assert fabric.argmax([3,1,9,2]) == 2
assert fabric.argmax([10,20,30]) == 2
assert fabric.argmax([5]) == 0
print("argmax OK")

# mul
assert fabric.mul([1,2,3,4],[4,3,2,1]) == [4,6,6,4]
assert fabric.mul([2,3,4],[2,2,2]) == [4,6,8]
print("mul OK")

# matmul_int8
assert fabric.matmul_int8([[1,2],[3,4]],[[1,0],[0,1]]) == [[1,2],[3,4]]
assert fabric.matmul_int8([[1,2],[3,4]],[[5,6],[7,8]]) == [[19,22],[43,50]]
print("matmul_int8 OK")

# biquad — coefficients in Q15 (multiply floats by 32768)
assert fabric.biquad([100,200,300,400], [32768,0,0,0,0]) == [100,200,300,400]
assert fabric.biquad([100,200,300,200,100], [10922,10922,10922,0,0]) == [33,99,199,233,199]  # verified on hardware
print("biquad OK")

# relu
assert fabric.relu([-5, 0, 3, -1, 7]) == [0, 0, 3, 0, 7]
assert fabric.relu([0]) == [0]
assert fabric.relu([1, 2, 3]) == [1, 2, 3]
print("relu OK")

# scale and add
assert fabric.scale([1, 2, 3, 4], 3) == [3, 6, 9, 12]
assert fabric.scale([0, -5, 7], -2) == [0, 10, -14]
assert fabric.add([1, 2, 3], [10, 20, 30]) == [11, 22, 33]
assert fabric.add([-1, 0, 1], [1, 0, -1]) == [0, 0, 0]
print("scale add OK")

# max_pool1d
assert fabric.max_pool1d([3,1,4,1,5,9,2,6,5,3,5,8], 3) == [4,9,6,8]
assert fabric.max_pool1d([10,20,5,15,3,8,12,7], 4) == [20,12]
print("max_pool1d OK")

# matmul
assert fabric.matmul([[1,2],[3,4]], [[5,6],[7,8]]) == [[19,22],[43,50]]
assert fabric.matmul([[1,0,2],[0,3,0]], [[1,2],[3,4],[5,6]]) == [[11,14],[9,12]]
print("matmul OK")

# clip
assert fabric.clip([-10,-5,0,5,10], -5, 5) == [-5,-5,0,5,5]
assert fabric.clip([127,-128,0], 0, 100) == [100,0,0]
assert fabric.clip([3,7,15,1], 4, 12) == [4,7,12,4]
print("clip OK")

# requantize
r = fabric.requantize([1000, -2000, 300, 0], 1, 4, 5)
assert r == [67, -120, 23, 5]
r = fabric.requantize([10000, -10000, 50], 2, 6, 0)
assert r == [127, -128, 1]
print("requantize OK")

# sum_squares and l2_norm
assert fabric.sum_squares([3, 4]) == 25
assert fabric.sum_squares([1, 2, 3, 4]) == 30
assert fabric.l2_norm([3, 4], 256) == [153, 204]
assert fabric.l2_norm([0, 5, 0, 0], 128) == [0, 128, 0, 0]
print("l2_norm OK")

# biquad_cascade (runs on scalar — cascade buffer ordering not Fabric-safe)
assert fabric.biquad_cascade([100,200,300], [[32768,0,0,0,0],[32768,0,0,0,0]]) == [100,200,300]
assert fabric.biquad_cascade([32768,32768,32768], [[16384,0,0,0,0],[16384,0,0,0,0]]) == [8192,8192,8192]
print("biquad_cascade OK")

# conv2d_int8
out = fabric.conv2d_int8([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],[1,0,0,1],4,4,2,2)
assert out[0] == 7 and out[4] == 17 and out[8] == 27
out2 = fabric.conv2d_int8([1,2,3,4,5,6,7,8,9],[1,1,1,1],3,3,2,2)
assert out2[0] == 12 and out2[3] == 28
print("conv2d_int8 OK")

# conv1d
r = fabric.conv1d([1,2,3,4,5,6,7,8,9,10],[1,0,-1,0,1,0,1,1,1,0,0,1],2,2,3)
assert r[0] == 0 and r[1] == 15
assert r[2] == 2 and r[3] == 23
assert r[4] == 4 and r[5] == 31
print("conv1d OK")

# avg_pool1d
assert fabric.avg_pool1d([3,1,4,1,5,9,2,6,5,3,5,8], 3) == [2,5,4,5]
assert fabric.avg_pool1d([10,20,5,15,3,8,12,7], 4) == [12,7]
assert fabric.avg_pool1d([0,10,20,30], 2) == [5,25]
assert fabric.avg_pool1d([-6,-4,0,8], 2) == [-5,4]
print("avg_pool1d OK")

# pointwise_conv
r1 = fabric.pointwise_conv([1,2,3,4], [1,0,0,0, 0,1,0,0, 1,1,1,1], 4, 3)
assert r1 == [1, 2, 10]
r2 = fabric.pointwise_conv([2,-1,3], [1,2,-1, -1,0,2], 3, 2)
assert r2 == [-3, 4]
print("pointwise_conv OK")

# threshold, vmax, vmin
assert fabric.threshold([3,-1,7,0,5], 3) == [0,0,1,0,1]
assert fabric.vmax([1,8,3,6], [5,2,3,4]) == [5,8,3,6]
assert fabric.vmin([1,8,3,6], [5,2,3,4]) == [1,2,3,4]
print("threshold vmax vmin OK")

print("ALL PASS")
