import numpy as np

# c = arrays = [np.random.randint(0, 5, 3)for _ in range(5)]
# d = arrays = [np.random.randint(10, 15, 3) for _ in range(5)]

c = np.array(np.random.randn(6, 5, 4, 3))
d = np.array(np.random.randn(6, 5, 4, 3))

# a = np.array(np.random.r(0, 3, 4) for _ in range(10) )

print('第一个数!!!!!!!!组：')
# print (c)
print(c.shape)
print('\n')
# b = np.array([[5,6],[7,8]])
#
print('第二个数组：')
# print (d)
print('\n')

print('沿轴 0 堆叠两个数组：')
# print (np.stack((c, d), 0))
print((np.stack((c, d), 0)).shape)
print('\n')

print('沿轴 1 堆叠两个数组：')
# print (np.stack((c, d),1))
print((np.stack((c, d), 1)).shape)
print('\n')

print('沿轴 2 堆叠两个数组：')
# print (np.stack((c, d), 2))
print((np.stack((c, d), 2)).shape)
print((np.stack((c, d), 3)).shape)
print((np.stack((c, d), 4)).shape)
