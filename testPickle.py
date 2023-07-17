import numpy as np
# from GP_regression import scalar_para_RBF_kernel
from GP_regression import Gaussian_Process_Regression
from GP_regression import DGram_optimize
from GP_regression import Error_optimize
from GP_regression import ConEn_optimize
from GP_regression import UCB_optimize

import scipy.io as scio
from scipy.spatial.transform import Rotation as R
from scipy import signal
import random
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as MSE

import pickle

X_test = np.load('X_test.npy')
Y_test = np.load('Y_test.npy')

print('X_test,',X_test)

with open('learnedGPR.pkl', 'rb') as f:
    # Call pickle.load to deserialize the object
    gpr = pickle.load(f)

print('pickle readed,')
print('np.array([X_test[0,:]])',np.array([X_test[0,:]]))

mu_single = gpr.predict(np.array([X_test[0,:]]))
print('mu_single: ', mu_single)

[mu, cov] = gpr.predict(X_test)
_,dim = np.shape(X_test)
# cov[cov<0] = 0
# print('test_y:',test_y)
# print('cov:',cov.flatten().tolist())

plt.figure()
for dim_I in range(dim):
    if dim_I > 8:
        plt.plot(X_test[:,dim_I], label=str(dim_I))
plt.legend()
plt.show()

plt.figure()

print('shape mu', np.shape(mu))
print('shape cov', np.shape(cov))
uncertainty = 1.96 *  np.array(np.sqrt(np.diag(cov)))

predict_y_s = []
true_y_s = []

for i in range(6):
    predict_y=np.array(mu[:,i].ravel().tolist()[0])
    true_y   =Y_test[:,i].ravel()

    true_y_s. append(true_y)
    predict_y_s. append(predict_y)



    error_y  = predict_y - true_y

    print('error_y:', error_y)
    print('error_y shape:', np.shape(error_y))
    print('error_y type:', type(error_y))

    print('MSE:', MSE(predict_y,true_y ))
    print('Mean uncertainty:', np.mean(uncertainty ))
    # fig, ax = plt.subplots()
    plt.subplot(6, 1, i+1)
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # plt.title("plot "+str(i+1))
    time =  np.linspace(start = 1, stop = 20, num = 20000)

    plt.plot(time,predict_y)
    plt.plot(time,true_y)

    plt.fill_between(time, predict_y + uncertainty, predict_y - uncertainty, alpha=0.5)
    # plt.legend( loc='upper right')
plt.show()