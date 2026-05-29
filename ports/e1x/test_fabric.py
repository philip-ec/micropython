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

print("ALL PASS")
