from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt

# apple = np.array([155, 156, 157])
# n = len(apple)

# model = LinearRegression().fit(np.arange(n).reshape((n, 1)), apple)

# print(model.predict([[3], [4]]))

n = 10000
x = np.linspace(-2 * np.pi, 2 * np.pi, n)
target = np.sin(x)

model = svm.SVR(C=10, epsilon=0.0001)
model.fit(x.reshape(n, 1), target)

n = 1000
x_test = np.linspace(- 2 * np.pi, 5 * np.pi, n)
x_test = x_test.reshape(n, 1)

# plt.plot(x, np.sin(x), "b,", label="Training data")
# plt.plot(x_test, model.predict(x_test), "r,", label="Machine learning estimate")
plt.plot(x_test, np.sin(x_test), "g", label="Real")

# print(model.predict([[2 * np.pi], [3 * np.pi / 2]]))

# plt.legend()
# plt.show()

n = 1000
x = np.linspace(-2 * np.pi, 2 * np.pi, n)
target = np.sin(x)

model2 = svm.SVR(C=10, epsilon=0.0001)
model2.fit(x.reshape(n, 1), target)
plt.plot(x_test, model.predict(x_test), ".", label="10 000 entrainements")
plt.plot(x_test, model2.predict(x_test), ".", label="1 000 entrainements")

plt.legend()
plt.show()
