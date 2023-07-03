import numpy as np
import logging
import random
from scipy.spatial.transform import Rotation as R

class scalar_para_von_mises_RBF_kernel:
    def __init__(self, feature_num, default_theta = 2e-2):
        self. theta = default_theta
    def compute( self, input1, input2):
         Rot_1 = R.from_euler('zyx', input1[0:3], degrees=False)
         Rot_2 = R.from_euler('zyx', input2[0:3], degrees=False)
         First_part = self. theta * np.exp( np.trace( Rot_1.as_matrix().T * Rot_2.as_matrix()) )
         Second_part = self. theta * np.exp(-   pow( np.linalg.norm((input1[3:] - input2[3:])), 2) )
         return (First_part + Second_part)

class scalar_para_von_mises_RBF_kernel_with_9_rot_vec:
    def __init__(self, feature_num, default_theta_Rot = 0.5, default_theta_Vec = 0.5, default_l = 0.7):
        self. sigma1 = default_theta_Rot
        self. sigma2 = default_theta_Vec
        self. l      = default_l
    def compute( self, input1, input2):
        #  exp_in = np.dot(input1[0:9],input2[0:9])
        #  First_part = self. theta1 * np.exp( exp_in )
        #  Second_part = self. theta2 * np.exp(-   np.linalg.norm((input1[9:] - input2[9:])**2) )
        #  return (First_part + Second_part)


        print('shape_of_input1:', np.shape(input1))
        print('shape_of_input1:', np.shape(input2))

        fst_input1 = input1[:,0:9]
        fst_input2 = input2[:,0:9]

        mul_matrix = np.matmul(fst_input1, fst_input2.T)

        fst_mat = self. sigma1  ** 2 * np.exp( mul_matrix)

        print('shape_of_mul_matrix:', np.shape(mul_matrix))

        sec_input1 = input1[:,9:]
        sec_input2 = input2[:,9:]

        

        dist_matrix = np.sum(sec_input1**2, 1).reshape(-1, 1) + np.sum(sec_input2**2, 1) - 2 * np.dot(sec_input1, sec_input2.T)

        print('shape_of_dist_matrix:', np.shape(dist_matrix))

        sec_mat = self. sigma2  ** 2 * np.exp(- 0.5 / self. l ** 2 * dist_matrix)
        return  np.multiply(fst_mat, sec_mat)


class square_exponential_kernel:
    def __init__(self, feature_num, default_sigma_f = 0.5, default_l = 0.5):
        self. sigma = default_sigma_f
        self. l     = default_l
    def compute( self, input1, input2):
         dist_matrix = np.sum(input1**2, 1).reshape(-1, 1) + np.sum(input2**2, 1) - 2 * np.dot(input1, input2.T)
         return  self. sigma  ** 2 * np.exp(- 0.5 / self. l ** 2 * dist_matrix)

class Gaussian_Process_Regression:
        def __init__(self, kernel_name):
            self. kernel_class    = globals()[kernel_name]
            self. has_fitted = False

        def cache(self, train_X, train_Y):
            self. X = train_X
            self. Y = train_Y
            [Xraw, Xcol] = np.shape(self. X)
            [Yraw, Ycol] = np.shape(self. Y)
            self. output_dim = Ycol

            if Xraw != Yraw:
                logging.error("Data dimensions of input and output do not equal!!!")
            self. feature_num = Xraw
            self. feature_dim = Xcol
            self. kernel = self. kernel_class(self.feature_num )

        def fit(self, train_X, train_Y):
            self. X = train_X
            self. Y = train_Y
            # print('self. X', self. X)
            [Xraw, Xcol] = np.shape(self. X)
            [Yraw, Ycol] = np.shape(self. Y)
            self. output_dim = Ycol

            if Xraw != Yraw:
                logging.error("Data dimensions of input and output do not equal!!!")
            self. feature_num = Xraw
            self. feature_dim = Xcol

            self. kernel = self. kernel_class(self.feature_num )

            self. Gram_matrix =  self. kernel. compute(self. X, self. X)
            self. Gram_inv =np.linalg.pinv(np.mat( self. Gram_matrix) + 1e-10 * np.eye(self. Gram_matrix.shape[0]))
            print('shape_of_Gram_inv:', np.shape(self. Gram_inv))
            self. has_fitted = True

        
        def remove_feature_at(self, i ):
             self. X = np.delete(self. X, i, 0)
             self. Y = np.delete(self. Y, i, 0)
             self.fit(self. X, self. Y)
            
        def predict(self, input_X):
            [Xraw, Xcol] = np.shape(input_X)
            if Xcol != self. feature_dim:
                  logging.error("Prediction input dimensions of input do not equal to training!!!")
            # print('Xraw',Xraw)
            # print('self. output_dim',self. output_dim)

            Kzz = self.kernel. compute(input_X, input_X)
            KXz = self.kernel. compute(self.X, input_X)


            mu_predict = KXz.T.dot(self. Gram_inv).dot(self.Y)
            cov_predict = Kzz - KXz.T.dot(self. Gram_inv).dot(KXz)

            cov_predict[cov_predict<0] = 0
            return (mu_predict, cov_predict)

                 # [feature_num] *  [feature_num X feature_num] * [feature_num X output_dim] = [1 X output_dim]

def Gram_optimize(gpr:Gaussian_Process_Regression, target_num, compare_seed_num):
    critics_index = np.zeros(gpr. feature_num) 
    for k in range(gpr. feature_num):
        ran = random.sample(range(gpr. feature_num),compare_seed_num)
        Gram_matrix =np.mat(np.zeros([compare_seed_num + 1, compare_seed_num + 1]))
        X_test = gpr.X[ran,:]
        X_test_plus_here = np.row_stack((X_test, np.array([gpr.X[k,:]])))
        # print('X_test_plus_here_shape:', np.shape(X_test_plus_here))
        for i in range( compare_seed_num + 1 ):
            for j in range( compare_seed_num + 1 ):
                if i <= j:
                    Gram_matrix[i,j] = gpr. kernel. compute(X_test_plus_here[i], X_test_plus_here[j])
                else:
                    Gram_matrix[i,j] = Gram_matrix[j,i]
        
        critics_index[k] = np.linalg.det(Gram_matrix)
        

        print('Gram_optimize process', 100 * (k+1) / gpr. feature_num, '%')

    critic_ind_sort =  np.argsort(critics_index)
    choose_sort = critic_ind_sort[(gpr. feature_num - target_num) : gpr. feature_num]

    X_data = gpr.X[choose_sort,:]
    Y_data = gpr.Y[choose_sort,:]
    gpr.fit(X_data, Y_data)

    return gpr

def PI_Bayes_optimize(gpr:Gaussian_Process_Regression, target_num, initial_num):
    serials = random.sample(range(gpr. feature_num),initial_num)
    X_data = gpr.X[serials,:]
    Y_data = gpr.Y[serials,:]


    print('X_data_shape:',np.shape(X_data))
    print('Y_data_shape:',np.shape(Y_data))

    remain_data_serial = set([i for i in range(gpr.feature_num)]) - set(serials)
    # cov_predict = np.zeros(target_num - initial_num)

    for k in range(target_num - initial_num):
        Gram_matrix =np.mat(np.zeros([initial_num + k, initial_num + k]))
        for i in range( initial_num + k ):
            for j in range( initial_num + k ):
                if i <= j:
                    Gram_matrix[i,j] = gpr. kernel. compute(X_data[i], X_data[j])
                    if Gram_matrix[i,j] < 0:
                        Gram_matrix[i,j] = 0
                else:
                    Gram_matrix[i,j] = Gram_matrix[j,i]
        Gram_matrix_inv = np.linalg.pinv(np.mat( Gram_matrix))
        
        list_remain_data_serial = list(remain_data_serial)
        PI = [0.0] * len(list_remain_data_serial)

        for i in range(len(list_remain_data_serial)):
            K_zX =  [0.0] * (initial_num + k)
            for j in range( initial_num + k ):   
                K_zX[j] = gpr. kernel. compute(gpr.X[list_remain_data_serial[i]], X_data[j])
            K_zz = gpr. kernel. compute(gpr.X[list_remain_data_serial[i]],\
                                         gpr.X[list_remain_data_serial[i]])
            mu_predict = np.mat(K_zX).dot(Gram_matrix_inv).dot( np.mat(Y_data))
            mat_cov = K_zz - np.mat(K_zX).dot(Gram_matrix_inv).dot(np.mat(K_zX).T)
            cov_predict = mat_cov[0,0]
            # print('mu_predict - np.mat(Y_data)', mu_predict - gpr.Y[list_remain_data_serial[i]])
            distance = np.linalg.norm( np.array(mu_predict) -np.array(gpr.Y[list_remain_data_serial[i]]))
            PI[i] = np.exp( 0.1 + distance**2 )

        
        kth_select = PI.index(max(PI))
        print('kth_select: ',kth_select)
        remain_data_serial = remain_data_serial - set([kth_select])
        
        
        # print(' gpr.X[list_remain_data_serial[kth_select]]:', gpr.X[list_remain_data_serial[kth_select]])

        print('X_data_shape:',np.shape(X_data))
        # print('gpr.X_inserted_shape:',np.shape(gpr.X[list_remain_data_serial[kth_select]]))

        X_data = np.row_stack((X_data, gpr.X[list_remain_data_serial[kth_select]]))
        Y_data = np.row_stack((Y_data, gpr.Y[list_remain_data_serial[kth_select]]))


        print('Bayes_optimize process', 100 * (k+1) / (target_num - initial_num), '%')

    
    gpr.fit(X_data, Y_data)

    return gpr

def Exp_optimize(gpr:Gaussian_Process_Regression, target_num, initial_num):
    from scipy.stats import norm
    serials = random.sample(range(gpr. feature_num),initial_num)
    X_data = gpr.X[serials,:]
    Y_data = gpr.Y[serials,:]


    print('X_data_shape:',np.shape(X_data))
    print('Y_data_shape:',np.shape(Y_data))

    remain_data_serial = set([i for i in range(gpr.feature_num)]) - set(serials)
    # exp_critics = np.zeros(target_num - initial_num)

    for k in range(target_num - initial_num):
        Gram_matrix =np.mat(np.zeros([initial_num + k, initial_num + k]))
        for i in range( initial_num + k ):
            for j in range( initial_num + k ):
                if i <= j:
                    Gram_matrix[i,j] = gpr. kernel. compute(X_data[i], X_data[j])
                else:
                    Gram_matrix[i,j] = Gram_matrix[j,i]

        
        list_remain_data_serial = list(remain_data_serial)
        # exp_critics = np.zeros(target_num - initial_num)
        exp_critics = [0.0] * len(list_remain_data_serial)
        Gram_matrix_inv = np.linalg.pinv(np.mat( Gram_matrix))

        for i in range(len(list_remain_data_serial)):
            K_zX =  [0.0] * (initial_num + k)
            for j in range( initial_num + k ):   
                K_zX[j] = gpr. kernel. compute(gpr.X[list_remain_data_serial[i]], X_data[j])
            K_zz = gpr. kernel. compute(gpr.X[list_remain_data_serial[i]],\
                                         gpr.X[list_remain_data_serial[i]])
            mu_predict = np.mat(K_zX).dot(Gram_matrix_inv).dot( np.mat(Y_data))
            mat_cov = K_zz - np.mat(K_zX).dot(Gram_matrix_inv).dot(np.mat(K_zX).T)
            delta_predict = np.sqrt(mat_cov[0,0])
            if delta_predict <= 0:
                exp_critics[i] = 0
            else:
                epsilon = 0
                Z = (np.max(mu_predict - np.array(gpr.Y[list_remain_data_serial[i]])) - epsilon)/delta_predict
                # print('Z: ',Z)
                # print('mu_predict: ',np.array(mu_predict).ravel())
                # print('max:',np.max(mu_predict - np.array(gpr.Y[list_remain_data_serial[i]])) )
                exp_critics[i] =  np.max(mu_predict - np.array(gpr.Y[list_remain_data_serial[i]])) * \
                                norm.cdf(Z,loc=0,scale=1) + delta_predict * norm.pdf(Z,loc=0,scale=1) 

        # print('exp_critics:', exp_critics)
        kth_select = exp_critics.index(max(exp_critics))
        print('kth_select: ',kth_select)
        remain_data_serial = remain_data_serial - set([kth_select])
        
        
        print(' gpr.X[list_remain_data_serial[kth_select]]:', gpr.X[list_remain_data_serial[kth_select]])

        print('X_data_shape:',np.shape(X_data))
        print('gpr.X_inserted_shape:',np.shape(gpr.X[list_remain_data_serial[kth_select]]))

        X_data = np.row_stack((X_data, gpr.X[list_remain_data_serial[kth_select]]))
        Y_data = np.row_stack((Y_data, gpr.Y[list_remain_data_serial[kth_select]]))


        print('EXP_optimize process', 100 * (k+1) / (target_num - initial_num), '%')

    
    gpr.fit(X_data, Y_data)

    return gpr
