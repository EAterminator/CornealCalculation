import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# 示例数据点（x, y）
x = np.array([3, 1, 0, 1])
y = np.array([1, 3, 1, 0])

# 定义残差函数
def residuals(c, x, y):
    """计算每个数据点到圆的距离与半径之差的平方"""
    ri = np.sqrt((x - c[0])**2 + (y - c[1])**2)
    return ri - c[2]

# 初始猜测
x0 = np.array([0, 0, 1])

# 使用最小二乘法拟合圆
res = least_squares(residuals, x0, args=(x, y))

# 拟合结果
a, b, r = res.x

# 绘制数据点和拟合的圆
circle = plt.Circle((a, b), r, color='blue', fill=False)
fig, ax = plt.subplots()
ax.add_artist(circle)
plt.scatter(x, y, color='red')
plt.axis('equal')
plt.show()

print(f"圆心: ({a}, {b}), 半径: {r}")