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
from sklearn.metrics import mean_squared_error as MSE

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
    


    # if i == 11:
    #     X_test = X_data
    #     Y_test = Y_data

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



ran = np.linspace(start = 0, stop = len(X_data_all)-1, num = 10000, dtype = int)

train_X = X_data_all[ran,:]
train_Y = Y_data_all[ran,:]


kernel_target_nums = [100, 250, 500, 1000, 1500, 2000, 2500, 3000]
MUS = []
MMS = []


for k_target in kernel_target_nums:
    gpr = Gaussian_Process_Regression('scalar_para_von_mises_RBF_kernel_with_9_rot_vec')
    gpr.cache(train_X, train_Y)
    step_num = int(k_target/10)

    # gpr = DGram_optimize(gpr, kernel_target_nums, step_num)
    # gpr = ConEn_optimize(gpr, kernel_target_nums, step_num, step_num)

    gpr = UCB_optimize(gpr, k_target, step_num, step_num)
    # gpr = Error_optimize(gpr, kernel_target_nums, step_num, step_num)

    [mu, cov] = gpr.predict(train_X)

    

    print('shape mu', np.shape(mu))
    print('shape cov', np.shape(cov))
    uncertainty = 1.96 *  np.array(np.sqrt(np.diag(cov)))

    MSEs = []

    for i in range(6):
        predict_y=np.array(mu[:,i].ravel().tolist()[0])
        true_y   =train_Y[:,i].ravel()

        error_y  = predict_y - true_y

        print('error_y:', error_y)
        print('error_y shape:', np.shape(error_y))
        print('error_y type:', type(error_y))

        print('MSE:', MSE(predict_y,true_y ))
        MSEs = MSEs + [MSE(predict_y,true_y )]

    MUS = MUS + [np.mean(uncertainty )]
    MMS = MMS + [np.mean(MSEs)]
    print('Mean uncertainty:', np.mean(uncertainty ))
    print('mean of MSEs:', np.mean(MSEs))

scio.savemat('USCB_learning_results', { 'kernel_target_nums':kernel_target_nums,\
                                        'MUS':MUS,
                                        'MMS':MMS})
