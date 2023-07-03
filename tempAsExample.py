import numpy as np
from GP_regression import RBF_kernel
from GP_regression import Gaussian_Process_Regression
import scipy.io as scio

def gnerate_rate_with_time_step(input_list, time_step):
    rate = np.zeros( np.shape( input_list ) )
    rate[1 : -2, :] = (input_list[2 : -1, :] -  input_list[0 : -3, :]) / time_step /2
    return rate

TIME_STEP = 1e-3

## Data loading
Dir = 'F:\\PythonProject\\GaussianProcess\\Gpytests\\'
Experiment_name_list = ['5th_proposed', '5th_UT2', '5th_UT3']
Task_list = ['_C_0_1_0', '_C_1_0_0', '_L_1_0_0', '_P_1_0_0']
Rear_name = '_Traj_Tracking_File.mat'

Ex_num = len(Experiment_name_list)
Ta_num = len(Task_list)

path = Dir + Experiment_name_list[0] + Task_list[0] + Rear_name
Data =  scio.loadmat(path)
print('Dara.keys: ', Data.keys())


Total_body_rotation_list =  Data['Total_body_rotation_list']
Total_body_translation_list =  Data['Total_body_translation_list']

Total_Freq_stroke        =  Data['Total_Freq_stroke']
Total_pitch_input        =  Data['Total_pitch_input']
Total_roll_input         =  Data['Total_roll_input']
Total_yaw_input          =  Data['Total_yaw_input']

Total_Angular_velocity_filtered_list = Data['Total_Angular_velocity_filtered_list']
Total_Angular_velocity_filtered_vec  = Total_Angular_velocity_filtered_list[:,:,0]
Total_Angular_acc_list = gnerate_rate_with_time_step(Total_Angular_velocity_filtered_vec, TIME_STEP)
print('Total_body_rotation_list shape:', np.shape(Total_body_rotation_list))
print('Total_body_translation_list shape:', np.shape(Total_body_translation_list))
print('Total_Freq_stroke shape:', np.shape(Total_Freq_stroke))
print('Total_pitch_input shape:', np.shape(Total_pitch_input))
print('Total_roll_input shape:', np.shape(Total_roll_input))
print('Total_yaw_input shape:', np.shape(Total_yaw_input))
print('Total_Angular_velocity_filtered_vec shape:', np.shape(Total_Angular_velocity_filtered_vec))

print('Total_Angular_acc_list: ', Total_Angular_acc_list)

# for i in range(Ex_num):
#     for j in range(Ta_num):
#         path = Dir + Experiment_name_list[i] + Task_list[j] + Rear_name
#         print('path',path)
#         Data =  scio.loadmat(path)




def y(x, noise_sigma=0.0):
    x = np.asarray(x)
    y = np.cos(x) + np.random.normal(0, noise_sigma, size=x.shape)
    
    return y

train_X = np.array([3, 1, 4, 5, 9]).reshape(-1, 1)
train_y = y(train_X, noise_sigma=1e-4)

train_YY = np.concatenate((train_y,-train_y),axis=1)
print('train_y:',train_y)
print('train_YY:',train_YY)

test_X = np.arange(0, 10, 0.1).reshape(-1, 1)

gpr = Gaussian_Process_Regression('RBF_kernel')
gpr.fit(train_X, train_YY)
# gpr.remove_feature_at([0,1,3])
[mu, cov] = gpr.predict(test_X)
test_y =mu[:,1].ravel()
# cov[cov<0] = 0
print('test_y:',test_y)
print('cov:',cov.flatten().tolist())
cov_l = cov.flatten().tolist()

uncertainty = 1.96 * np.sqrt(cov_l)

# print('uncertainty:',uncertainty)

import matplotlib.pyplot as plt
plt.figure()
plt.fill_between(test_X.ravel(), test_y + uncertainty, test_y - uncertainty, alpha=0.1)
plt.plot(test_X, test_y, label="predict")
plt.scatter(train_X, train_YY[:,1], label="train", c="red", marker="x")
plt.legend()
plt.show()