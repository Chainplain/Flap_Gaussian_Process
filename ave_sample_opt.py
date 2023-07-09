import numpy as np
# from GP_regression import scalar_para_RBF_kernel
from GP_regression import Gaussian_Process_Regression
from GP_regression import DGram_optimize
from GP_regression import PI_Bayes_optimize
from GP_regression import Error_optimize
from GP_regression import ConEn_optimize

import scipy.io as scio
from scipy.spatial.transform import Rotation as R
from scipy import signal
import random
import matplotlib.pyplot as plt


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
Dir = 'F:\\PythonProject\\GaussianProcess\\Gpytests\\'
Experiment_name_list = ['5th_proposed', '5th_UT2', '5th_UT3']
Task_list = ['_C_0_1_0', '_C_1_0_0', '_L_1_0_0', '_P_1_0_0']
Rear_name = '_Traj_Tracking_File.mat'

Ex_num = len(Experiment_name_list)
Ta_num = len(Task_list)
print('Ex_num: ',Ex_num)
print('Ta_num: ',Ta_num)

# path = Dir + Experiment_name_list[1] + Task_list[1] + Rear_name
# print('path: ', path)

FirstFlag = True

fb, fa = signal.butter(3, 1e-2)

for i in range(Ex_num):
    for j in range(Ta_num):
        path = Dir + Experiment_name_list[i] + Task_list[j] + Rear_name
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
                np.matmul(Total_body_rotation_list[kk,:,:].T, Total_translation_acc_list[kk,:].T).T
            Total_body_rotation_list_in_vec[kk,:] = np.asarray(Total_body_rotation_list[kk,:,:]).ravel()

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
        


        if i == 2 and j == 2:
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



ran = np.linspace(start = 0, stop = len(X_data_all)-1, num = 2000, dtype = int)

train_X = X_data_all[ran,:]
train_Y = Y_data_all[ran,:]



gpr = Gaussian_Process_Regression('scalar_para_von_mises_RBF_kernel_with_9_rot_vec')
gpr.fit(train_X, train_Y)
# gpr = DGram_optimize(gpr, 1000, 100)
# gpr = ConEn_optimize(gpr, 500, 100, 100)
gpr = Error_optimize(gpr, 1000, 100, 100)
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

for i in range(6):
    test_y=np.array(mu[:,i].ravel().tolist()[0])
    print('test_y:',test_y)
    plt.subplot(6, 1, i+1)
    plt.title("plot "+str(i+1))
    time =  np.linspace(start = 1, stop = 20, num = 20000)

    plt.plot(time,test_y, label="Estimated signal")
    plt.plot(time,Y_test[:,i].ravel(), label="Expected signal")
    
    plt.fill_between(time, test_y + uncertainty, test_y - uncertainty, alpha=0.5)
    plt.legend( loc='upper right')
plt.show()
print('uncertainty:',uncertainty)

