# 特征包括第一列房子面积，第二列房间个数；第三列为价格。

# 读入数据
import numpy as np

f = open("data.txt", 'r')
house_size = []
room_num = []
price = []
for i in f.readlines():
    house_size.append(float(i.split(',')[0]))
    room_num.append(float(i.split(',')[1]))
    price.append(float(i.split(',')[2].strip('/n')))

# 归一化
x1 = np.array(house_size).reshape(-1, 1)
x2 = np.array(room_num).reshape(-1, 1)
y = np.array(price).reshape(-1, 1)
data = np.concatenate((x1, x2, y), axis=1)

mean = np.mean(data, axis=0)  # 计算每一列的均值
ptp = np.ptp(data, axis=0)  # 计算每一列的最大最小差值
nor_data = (data - mean) / ptp  # 归一化
X = np.insert(nor_data[..., :2], 0, 1, axis=1)  # 添加x0=1
Y = nor_data[..., -1]
print(X.shape)

def normal_equation(X, y):
    return np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)

print(normal_equation(X,Y))
