import numpy as np
import matplotlib.pyplot as plt
from GP_regression import square_exponential_kernel
from GP_regression import Gaussian_Process_Regression
def y(x, noise_sigma=0.0):
    x = np.asarray(x)
    y = np.cos(x) + np.random.normal(0, noise_sigma, size=x.shape)
    return y.tolist()

train_X = np.array([3, 1, 4, 5, 9]).reshape(-1, 1)
train_y = y(train_X, noise_sigma=1e-4)
test_X = np.arange(0, 10, 0.1).reshape(-1, 1)

gpr = Gaussian_Process_Regression('square_exponential_kernel')
gpr.fit(train_X, train_y)
mu, cov = gpr.predict(test_X)

test_y=np.array(mu.ravel().tolist()[0])


uncertainty = 1.96 *  np.array(np.sqrt(np.diag(cov)))

# print('cov:',cov)
# print('test_y:',(test_y))
# print('uncertainty:',(uncertainty))
# print('type_test_y:',type(test_y))
# print('type_uncertainty:',type(uncertainty))
# print('shape_test_y:',np.shape(test_y))
# print('shape_uncertainty:',np.shape(uncertainty))


plt.figure()
# plt.title("l=%.2f sigma_f=%.2f" % (gpr.params["l"], gpr.params["sigma_f"]))
plt.fill_between(test_X.ravel(), test_y + uncertainty, test_y - uncertainty, alpha=0.1)
plt.plot(test_X, test_y, label="predict")
plt.scatter(train_X, train_y, label="train", c="red", marker="x")
plt.legend()
plt.show()