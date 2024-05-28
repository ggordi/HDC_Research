from vector import Vector
import vector as vec

v1c1 = Vector()
v2c1 = Vector()
v3c1 = Vector()
sum1 = vec.consensus_sum([v1c1, v2c1, v3c1])
class1 = Vector()
b1 = vec.xor(sum1, class1)

v1c2 = Vector()
v2c2 = Vector()
v3c2 = Vector()
sum2 = vec.consensus_sum([v1c2, v2c2, v3c2])
class2 = Vector()
b2 = vec.xor(sum2, class2)

hil = vec.consensus_sum([b1, b2])

res = vec.xor(hil, v2c2)

print(vec.hamming_distance(res, class2))
