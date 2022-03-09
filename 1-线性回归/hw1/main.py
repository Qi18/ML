# task1:plotting the data
# 数据是一个txt文件，每一行存储一组数据，第一列数据为城市的人口，第二列数据城市饭店的利润。
# Part1:从txt文件中读取数据，绘制成散点图

import matplotlib.pyplot as plt
from sympy import symbols, diff

f = open("data.txt", 'r')
population = []
profit = []
num = 0  # 数集
for line in f.readlines():
    col1 = line.split(',')[0]
    col2 = line.split(',')[1].strip('/n')
    population.append(float(col1))
    profit.append(float(col2))
    num += 1

# plt.title("Scatter plot of training data")
# plt.xlabel("population of city")
# plt.ylabel("profit")
# plt.scatter(population, profit)
# plt.show()


# task2:梯度下降
def hypothesis_function(x, theta0, theta1):
    return theta0 + x * theta1


def cost_function(theta0, theta1):
    result = 0
    for i in range(num):
        result += 1.0 / (2 * num) * pow(hypothesis_function(population[i], theta0, theta1) - profit[i], 2)
    return result


theta = [0, 0]  # 参数初值
iteration = 1500  # 迭代次数
alpha = 0.01  # 学习率
iterations=[]
cost=[]
x, y, theta0, theta1 = symbols('x y theta0 theta1', real=True)
for i in range(iteration):
    temp0 = theta[0]
    temp1 = theta[1]
    temp0 = temp0 - alpha * diff(cost_function(theta0, theta1), theta0).subs({theta0: theta[0], theta1: theta[1]})
    temp1 = temp1 - alpha * diff(cost_function(theta0, theta1), theta1).subs({theta0: theta[0], theta1: theta[1]})
    theta[0] = temp0
    theta[1] = temp1
    iterations.append(i)
    cost.append(cost_function(theta[0],theta[1]))

# part3：绘制回归直线图，和损失函数变化图
x = [0.0, 22.5]
y = [0.0 * theta[1] + theta[0], 22.5 * theta[1] + theta[0]]
plt.plot(x, y, color="red")
plt.title("Linear Regression")
plt.xlabel("population of city")
plt.ylabel("profit")
plt.scatter(population, profit, marker='x')
plt.show()

plt.title("Visualizing J(θ)")
plt.xlabel("iterations")
plt.ylabel("cost")
plt.plot(iterations,cost, color="red")
plt.show()
