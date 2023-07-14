import numpy as np
# from GP_regression import scalar_para_RBF_kernel
from GP_regression import Gaussian_Process_Regression
from GP_regression import DGram_optimize
from GP_regression import PI_Bayes_optimize
from GP_regression import Error_optimize
from GP_regression import ConEn_optimize
from GP_regression import UCB_optimize

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
Dir = 'F:\\PythonProject\\GaussianProcess\\Gpytests-new\\5th_origin_'
Experiment_name_list = ['bC', 'C', 'L', 'mbC', 'mC', 'rbC', 'rC', 'rmC', 'sC', 'smallC', 'smallmC', 'bumping', 'bumpingm']
Rear_name = '__Traj_Tracking_File.mat'

Ex_num = len(Experiment_name_list)
print('Ex_num: ',Ex_num)

# path = Dir + Experiment_name_list[1] + Task_list[1] + Rear_name
# print('path: ', path)

FirstFlag = True

fb, fa = signal.butter(3, 1e-2)


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
    print('Total_translation_acc_list[kk,:].T:',Total_translation_acc_list[kk,:].T)
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
    print('----------------------------------------------- '+ Experiment_name_list[i])
    print('shape Total_Angular_velocity_filtered_vec_filtered: ',np.shape(Total_Angular_velocity_filtered_vec_filtered))
    print('shape Total_translation_vel_list_filtered_in_body: ',np.shape(Total_translation_vel_list_filtered_in_body))
    print('shape Total_Freq_stroke_filtered: ',np.shape(Total_Freq_stroke_filtered))
    print('shape Total_pitch_input_filtered: ',np.shape(Total_pitch_input_filtered))
    print('shape Total_roll_input_filtered: ',np.shape(Total_roll_input_filtered))
    print('shape Total_yaw_input_filtered: ',np.shape(Total_yaw_input_filtered))
    print('----------------------------------------------- ')
    print('mean Total_Angular_velocity_filtered_vec_filtered: ',np.mean(Total_Angular_velocity_filtered_vec_filtered,axis=0))
    print('mean Total_translation_vel_list_filtered_in_body: ',np.mean(Total_translation_vel_list_filtered_in_body,axis=0))
    print('mean Total_Freq_stroke_filtered: ',np.mean(Total_Freq_stroke_filtered))
    print('mean Total_pitch_input_filtered: ',np.mean(Total_pitch_input_filtered))
    print('mean Total_roll_input_filtered: ',np.mean(Total_roll_input_filtered))
    print('mean Total_yaw_input_filtered: ',np.mean(Total_yaw_input_filtered))
    print('----------------------------------------------- ')
    print('std Total_Angular_velocity_filtered_vec_filtered: ',np.std (Total_Angular_velocity_filtered_vec_filtered))
    print('std Total_translation_vel_list_filtered_in_body: ',np.std (Total_translation_vel_list_filtered_in_body))
    print('std Total_Freq_stroke_filtered: ',np.std (Total_Freq_stroke_filtered))
    print('std Total_pitch_input_filtered: ',np.std (Total_pitch_input_filtered))
    print('std Total_roll_input_filtered: ',np.std (Total_roll_input_filtered))
    print('std Total_yaw_input_filtered: ',np.std (Total_yaw_input_filtered))
    print('----------------------------------------------- ')