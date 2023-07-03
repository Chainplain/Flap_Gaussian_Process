import numpy as np


def target_generator(X, add_noise=False):
    target = 0.5 + np.sin(3 * X)
    if add_noise:
        rng = np.random.RandomState(1)
        target += rng.normal(0, 0.3, size=target.shape)
    return target.squeeze()

X = np.linspace(0, 5, num=30).reshape(-1, 1)
y = target_generator(X, add_noise=False)

import matplotlib.pyplot as plt

# plt.plot(X, y, label="Expected signal")
# plt.legend()
# plt.xlabel("X")
# _ = plt.ylabel("y")

rng = np.random.RandomState(0)
X_train = rng.uniform(0, 5, size=20).reshape(-1, 1)
y_train = np.array(target_generator(X_train, add_noise=True))
y_train =  np.array([y_train]).T
print('shape of y_train:', np.shape(y_train))
print(y_train)

from GP_regression import square_exponential_kernel
from GP_regression import Gaussian_Process_Regression

gpr = Gaussian_Process_Regression('square_exponential_kernel')
gpr.fit(X_train, y_train)


y_mean, y_std = gpr.predict(X)
print('Gram',gpr.Gram_matrix)
print('Gram_1',gpr.Gram_inv)



plt.plot(X, y, label="Expected signal")
plt.scatter(x=X_train[:, 0], y=y_train, color="black", alpha=0.4, label="Observations")

print('X',X)
print('y_mean',(y_mean.T.tolist())[0])
print('y_std',(y_std.T.tolist())[0])

plt.errorbar(X, ((y_mean.T).tolist())[0], ((y_std.T).tolist())[0])
plt.legend()
plt.xlabel("X")
plt.ylabel("y")


plt.show()
