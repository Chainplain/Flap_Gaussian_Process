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

def gnerate_rate_with_time_step(input_list, time_step):
    rate = np.zeros( np.shape( input_list ) )
    rate[1 : -2, :] = (input_list[2 : -1, :] -  input_list[0 : -3, :]) / time_step /2
    return rate

def rotation_matrix_2_euler_angle_in_rad(R_list):
    euler_list= np.zeros([len(R_list),3])
    for i in range(len(R_list)):
        Rot_m =  R.from_matrix(R_list[i,:,:])
        euler_list[i,:] = Rot_m. as_euler('zyx', degrees=False)
    return euler_list


TIME_STEP = 1e-3

## Data loading
Dir = 'F:\\PythonProject\\GaussianProcess\\Gpytests-new\\5th_origin_'
Experiment_name_list = ['bC', 'C', 'L', 'mbC', 'mC', 'rbC', 'rC', 'rmC', 'sC', 'smallC', 'smallmC', 'bumping', 'bumpingm']
Rear_name = '__Traj_Tracking_File.mat'

Ex_num = len(Experiment_name_list)
print('Ex_num: ',Ex_num)

# path = Dir + Experiment_name_list[1] + Task_list[1] + Rear_name
# print('path: ', path)

FirstFlag = True

fb, fa = signal.butter(3, 5e-3)


## Data preprocessing
for i in range(Ex_num):
    path = Dir + Experiment_name_list[i] + Rear_name
    Data =  scio.loadmat(path)
    print('path: ', path)


    Total_body_rotation_list =  Data['Total_body_rotation_list']
    Total_euler_angle_list   = rotation_matrix_2_euler_angle_in_rad(Total_body_rotation_list)

    Total_body_translation_list =  Data['Total_body_translation_list']

    Total_Freq_stroke        =  Data['Total_Freq_stroke'].T
    Total_pitch_input        =  Data['Total_pitch_input'].T
    Total_roll_input         =  Data['Total_roll_input'].T
    Total_yaw_input          =  Data['Total_yaw_input'].T

    Total_Angular_velocity_filtered_list = Data['Total_Angular_velocity_filtered_list']
    Total_Angular_velocity_filtered_vec  = Total_Angular_velocity_filtered_list[:,:,0]
    Total_Angular_acc_list = gnerate_rate_with_time_step(Total_Angular_velocity_filtered_vec, TIME_STEP)
    Total_translation_vel_list = gnerate_rate_with_time_step(Total_body_translation_list, TIME_STEP)
    # print('Total_body_translation_list:', Total_body_translation_list)
    
    Total_translation_vel_list_filtered = \
        np.zeros(np.shape(Total_translation_vel_list))
    
    for uu in range(3):
        Total_translation_vel_list_filtered[:,uu] = signal.filtfilt(fb, fa, Total_translation_vel_list[:,uu])

    # for kk in range(len(Total_translation_vel_list_filtered)):
    #     Total_translation_vel_list_filtered_in_body[kk,:] =  \
    #         np.matmul(Total_body_rotation_list[kk,:,:].T, Total_translation_vel_list_filtered[kk,:].T).T

    Total_translation_acc_list = gnerate_rate_with_time_step(Total_translation_vel_list_filtered, TIME_STEP)

    Total_translation_vel_list_filtered_in_body = np.zeros(np.shape(Total_translation_vel_list_filtered))
    Total_translation_acc_list_in_body = np.zeros(np.shape(Total_translation_acc_list))

    Total_body_rotation_list_in_vec =  np.zeros([len(Total_body_rotation_list), 9])


    for kk in range(len(Total_body_rotation_list)):
        Total_translation_vel_list_filtered_in_body[kk,:] =  \
            np.matmul(Total_body_rotation_list[kk,:,:].T, Total_translation_vel_list_filtered[kk,:].T).T
        Total_translation_acc_list_in_body[kk,:] =  \
            np.matmul(Total_body_rotation_list[kk,:,:].T, (Total_translation_acc_list[kk,:].T  +  np.array([0,0,9.8]))).T
        Total_body_rotation_list_in_vec[kk,:] = np.asarray(Total_body_rotation_list[kk,:,:]).ravel()
        # print('Total_translation_acc_list_in_body[kk,:]: ',Total_translation_acc_list_in_body[kk,:].T )
        # print('Total_translation_acc_list[kk,:]++: ',Total_translation_acc_list[kk,:].T  +  np.array([0,0,9.8]))

    # print('Total_body_rotation_list_in_vec',Total_body_rotation_list_in_vec[0,:])
    # print('Total_body_rotation_list_in_vec',Total_body_rotation_list_in_vec[666,:])

    Total_Angular_velocity_filtered_vec_filtered = \
        np.zeros(np.shape(Total_Angular_velocity_filtered_vec))
    # Total_translation_vel_list_in_body_filtered = \
    #     np.zeros(np.shape(Total_translation_vel_list_in_body))
    # Total_translation_acc_list_in_body_filtered = \
    #     np.zeros(np.shape(Total_translation_acc_list_in_body))
    Total_Angular_acc_list_filtered = \
        np.zeros(np.shape(Total_Angular_acc_list))
    for hh in range(3):
        # Total_translation_acc_list_in_body_filtered[:,hh] = signal.filtfilt(fb, fa, Total_translation_acc_list_in_body[:,hh])
        Total_Angular_acc_list_filtered[:,hh] = signal.filtfilt(fb, fa, Total_Angular_acc_list[:,hh])
        Total_Angular_velocity_filtered_vec_filtered[:,hh] = signal.filtfilt(fb, fa, Total_Angular_velocity_filtered_vec[:,hh])
        print('shape of  Total_Angular_acc_list_filtered[:,hh]', np.shape( Total_Angular_acc_list_filtered[:,hh]))
        # Total_translation_vel_list_in_body_filtered[:,hh] = signal.filtfilt(fb, fa, Total_translation_vel_list_in_body[:,hh])
    Total_pitch_input_filtered = \
        np.zeros(np.shape(Total_pitch_input))
    Total_roll_input_filtered = \
        np.zeros(np.shape(Total_roll_input))
    Total_yaw_input_filtered = \
        np.zeros(np.shape(Total_yaw_input))
    Total_Freq_stroke_filtered = \
        np.zeros(np.shape(Total_yaw_input))
    
    Total_pitch_input_filtered[:,0] = signal.filtfilt(fb, fa, Total_pitch_input[:,0])
    Total_roll_input_filtered[:,0] = signal.filtfilt(fb, fa, Total_roll_input[:,0])
    Total_yaw_input_filtered[:,0] = signal.filtfilt(fb, fa, Total_yaw_input[:,0])
    Total_Freq_stroke_filtered[:,0]= signal.filtfilt(fb, fa, Total_Freq_stroke[:,0])
    
    # print('----------------------------------------------- ')
    # print('mean Total_Angular_velocity_filtered_vec_filtered: ',np.mean(Total_Angular_velocity_filtered_vec_filtered))
    # print('mean Total_translation_vel_list_filtered_in_body: ',np.mean(Total_translation_vel_list_filtered_in_body))
    # print('mean Total_Freq_stroke_filtered: ',np.mean(Total_Freq_stroke_filtered))
    # print('mean Total_pitch_input_filtered: ',np.mean(Total_pitch_input_filtered))
    # print('mean Total_roll_input_filtered: ',np.mean(Total_roll_input_filtered))
    # print('mean Total_yaw_input_filtered: ',np.mean(Total_yaw_input_filtered))
    # print('----------------------------------------------- ')
    # print('std Total_Angular_velocity_filtered_vec_filtered: ',np.std (Total_Angular_velocity_filtered_vec_filtered))
    # print('std Total_translation_vel_list_filtered_in_body: ',np.std (Total_translation_vel_list_filtered_in_body))
    # print('std Total_Freq_stroke_filtered: ',np.std (Total_Freq_stroke_filtered))
    # print('std Total_pitch_input_filtered: ',np.std (Total_pitch_input_filtered))
    # print('std Total_roll_input_filtered: ',np.std (Total_roll_input_filtered))
    # print('std Total_yaw_input_filtered: ',np.std (Total_yaw_input_filtered))
    # print('----------------------------------------------- ')

    mean_freq = np.mean(Total_Freq_stroke_filtered)
    X_data = np.concatenate((Total_body_rotation_list_in_vec,\
                            Total_Angular_velocity_filtered_vec_filtered,\
                            Total_translation_vel_list_filtered_in_body,\
                            Total_Freq_stroke_filtered - mean_freq,\
                            Total_pitch_input_filtered,\
                            Total_roll_input_filtered,\
                            Total_yaw_input_filtered\
                                ),axis=1)
    Y_data = np.concatenate((Total_translation_acc_list_in_body,\
                            Total_Angular_acc_list_filtered),axis=1)
    


    if i == 10:
        X_test = X_data
        Y_test = Y_data

    # else:
    if (FirstFlag):
        X_data_all = X_data
        Y_data_all = Y_data
        FirstFlag = False
    else:
        X_data_all = np.concatenate((X_data_all,X_data),axis=0)
        Y_data_all = np.concatenate((Y_data_all,Y_data),axis=0)
    print('X_data_all shape:', np.shape(X_data_all))
    print('Y_data_all shape:', np.shape(Y_data_all))




_,dim = np.shape(X_data)
print('dim:',dim)



ran = np.linspace(start = 0, stop = len(X_data_all)-1, num = 5000, dtype = int)

train_X = X_data_all[ran,:]
train_Y = Y_data_all[ran,:]



gpr = Gaussian_Process_Regression('scalar_para_von_mises_RBF_kernel_with_9_rot_vec')
gpr.cache(train_X, train_Y)
# gpr = DGram_optimize(gpr, 1000, 100)
# gpr = ConEn_optimize(gpr, 200, 50, 50)

gpr = UCB_optimize(gpr, 2000, 200, 200)
# gpr = Error_optimize(gpr, 1000, 100, 100)
# gpr.remove_feature_at([0,1,3])
[mu, cov] = gpr.predict(X_test)

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
# print('uncertainty:',uncertainty)


fn = 'learnedGPR.pkl'
with open(fn, 'wb') as f:
    pickle.dump(gpr, f)
np.save('X_test', X_test)
np.save('Y_test', Y_test)
# scio.savemat('show###.mat', {'true_y_s':true_y_s,\
#                           'predict_y_s':predict_y_s,
#                           'uncertainty':uncertainty})