import numpy as np
from GP_regression import scalar_para_RBF_kernel
from GP_regression import Gaussian_Process_Regression
from GP_regression import Gram_optimize
from GP_regression import PI_Bayes_optimize
from GP_regression import Exp_optimize


def target_generator(X, add_noise=False):
    target = 0.5 + np.sin(3 * X)
    if add_noise:
        rng = np.random.RandomState(1)
        target += rng.normal(0, 0.3, size=target.shape)
    return target.squeeze()

X = np.linspace(0, 5, num=30).reshape(-1, 1)
y = target_generator(X, add_noise=False)

import matplotlib.pyplot as plt

plt.plot(X, y, label="Expected signal")
plt.legend()
plt.xlabel("X")
_ = plt.ylabel("y")

rng = np.random.RandomState(0)
X_train = rng.uniform(0, 5, size=20).reshape(-1, 1)
y_train = target_generator(X_train, add_noise=True)
y_train = np.array([y_train]).T

print('X_train:',X_train)
print('y_train:',y_train)
print('X_train_type:',type(X_train))
print('y_train_type:',type(y_train))

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# kernel = 1.0 * RBF(length_scale=1e-1, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(
#     noise_level=1e-2, noise_level_bounds=(1e-10, 1e1)
# )
gpr = Gaussian_Process_Regression('scalar_para_RBF_kernel')
gpr.fit(X_train, y_train)
# print('kernel theta before:',kernel.theta)

# print('kernel theta after:',gpr.kernel_.theta)

# print('Feature number after:',gpr.n_features_in_)
print('Fit completed')

[y_mean, y_std] = gpr.predict(X)

print('X:',X)
print('y:',y)
print('y_mean:',y_mean)
print('y_std:',y_std)

print('predict')

plt.plot(X, y, label="Expected signal")
plt.plot(X, y_mean, label=" signal")
# plt.scatter(x=X_train[:, 0], y=y_train, color="black", alpha=0.4, label="Observations")
# plt.errorbar(X, y_mean, y_std)
# plt.legend()
# plt.xlabel("X")
# plt.ylabel("y")
# _ = plt.title(
#     f"Initial: {kernel}\nOptimum: {gpr.kernel_}\nLog-Marginal-Likelihood: "
#     f"{gpr.log_marginal_likelihood(gpr.kernel_.theta)}",
#     fontsize=8,
# )

plt.show()