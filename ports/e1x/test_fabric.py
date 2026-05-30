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

print("ALL PASS")
