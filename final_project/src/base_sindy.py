import pandas as pd
import numpy as np
import time
import mlflow
import math
import sympy
#from itertools import combinations
import os
from sklearn.linear_model import Ridge, Lasso
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

file_path = os.path.realpath(__file__)
print(file_path)
file_dir = '/'.join(file_path.split('/')[:-1])



class BaseSINDy(object):

    def __init__(self, config, poly_order=5,
                    #max_derivatives_order=4,
                    include_t=False,
                    include_poly_predictor=False, 
                    order_predict=1, 
                    thresh=1e-6, 
                    alpha=1.0, 
                    use_preprocessing = True,
                    k_spline = 3,
                    spline_method = None,
                    interpolated_dt = 0.1,
                    use_mlflow = True,
                    experiment_name='sindy_model', 
                    run_name='sindy_base') -> None:
        self.config = config
        self.poly_order = poly_order
        #self.max_derivatives_order = max_derivatives_order
        self.include_t = include_t
        self.include_poly_predictor = include_poly_predictor
        self.order_predict = order_predict
        self.thresh = thresh
        self.alpha = alpha
        self.use_preprocessing = use_preprocessing
        self.k_spline = k_spline
        self.spline_method = spline_method
        self.interpolated_dt = interpolated_dt
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.use_mlflow = use_mlflow
        self.x_splines = None
        if self.use_mlflow:
            mlflow.log_param('poly_order', poly_order)
            #mlflow.log_param('max_derivatives_order', max_derivatives_order)
            mlflow.log_param('include_t', include_t)
            mlflow.log_param('include_poly_predictor', include_poly_predictor)
            mlflow.log_param('thresh', thresh)
            mlflow.log_param('order_predict', order_predict)
            mlflow.log_param('alpha', alpha)
            mlflow.log_param('use_preprocessing', use_preprocessing)
            if use_preprocessing:
                mlflow.log_param('k_spline', k_spline)
                mlflow.log_param('spline_method', spline_method)
                mlflow.log_param('interpolated_dt', interpolated_dt)


    def get_interpolated_t(self, t):
        if self.interpolated_dt < 1:
            t_s = np.array(list(np.linspace(t[0], t[-1], int((t[-1] - t[0])//self.interpolated_dt))))
        elif self.interpolated_dt > 1:
            t_s = np.array(list(np.linspace(t[0], t[-1], int(self.interpolated_dt))))
        else:
            t_s = t
        return t_s
    
    def smooth(self, t, x, t_s=None, coef=None):
        if coef is None:
            coef=self.config['smooth_coef']
        spl = UnivariateSpline(t, x)
        spl.set_smoothing_factor(coef)
        #print('smoothing:', coef)
        if t_s is None:
            return t, spl(t)
        else:
            return t_s, spl(t_s)
    
    def preprocesser(self, t, X, n=None):
        print('\t\t start spline...')
        t_s = self.get_interpolated_t(t)
        self.x_splines = {}
        X_new = np.zeros((len(X), len(t_s)))
        for i, x in enumerate(X):
            t, x = self.smooth(t, x)
            start = time.time()
            spline = K_Poly_Spline(t, x, self.k_spline, self.spline_method)
            #print(list(spline.f.keys())[-1])
            end = start - time.time()
            X_new[i] = spline.interpolate(t_s)
            if self.use_mlflow:
                if n is None:
                    mlflow.log_artifact(f'preprocessing/spline_x_{i}', spline.f)
                else:
                    mlflow.log_artifact(f'preprocessing/spline_x_{i}_model_{n}', spline.f)
                mlflow.log_metric(f'preprocessing/x_{i}_solve_time', end)
            self.x_splines[i] = spline
        print('\t\t spline done.')
        return t_s, X_new

    def take_first_derivative(self, t, x, one_pt=True):
        if one_pt:
            dx = np.diff(x)/np.diff(t) 
            t, dx = self.smooth(t[:-1], dx, coef=self.config['derivative_smooth_coef'])
        else:
            dx = (-1*x[4:]+8*x[3:-1]-8*x[1:-3]+x[:-4])/(12*np.diff(t)[1:-2])
            t, dx = self.smooth(t[1:-3], dx, coef=self.config['derivative_smooth_coef'])
        return dx


    def take_second_derivative(self, t, x, one_pt=True):
        if one_pt:
            dxx = np.diff(self.take_first_derivative(t,x))/np.diff(t[:-1])
            t, dxx = self.smooth(t[:-2], dxx, coef=self.config['derivative_smooth_coef'])
        else:
            dxx = (-1*x[4:]+16*x[3:-1]-30*x[2:-2]+16*x[1:-3]-x[:-4])/(12*np.diff(t)[1:-2]**2)
            t, dxx = self.smooth(t[1:-3], dxx, coef=self.config['derivative_smooth_coef'])
        return dxx

    def take_third_derivative(self, t, x, one_pt=True):
        #return (x[4:] - 2*x[3:-1] + 2*x[1:-3] - x[:-4])/(2*np.diff(t)[1:-2]**3)
        if one_pt:
            dxxx = np.diff(self.take_second_derivative(t,x))/np.diff(t[:-2])
            t, dxxx = self.smooth(t[:-3], dxxx, coef=self.config['derivative_smooth_coef'])
        else:
            dxxx = (x[4:] - 2*x[3:-1] + 2*x[1:-3] - x[:-4])/(2*np.diff(t)[1:-2]**3)
            t, dxxx = self.smooth(t[1:-3], dxxx, coef=self.config['derivative_smooth_coef'])
        return dxxx

    def take_forth_derivative(self, t, x, one_pt=True):
        #return (x[4:] - 4*x[3:-1] + 6*x[2:-2] - 4*x[1:-3] + x[:-4])/(np.diff(t)[1:-2]**4)
        if one_pt:
            dxxxx = np.diff(self.take_third_derivative(t,x))/np.diff(t[:-3])
            t, dxxxx = self.smooth(t[:-4], dxxxx, coef=self.config['derivative_smooth_coef'])
        else:
            dxxxx = (x[4:] - 4*x[3:-1] + 6*x[2:-2] - 4*x[1:-3] + x[:-4])/(np.diff(t)[1:-2]**4)
            t, dxxxx = self.smooth(t[1:-3], dxxxx, coef=self.config['derivative_smooth_coef'])
        return dxxxx

    def take_derivative(self, t, x, derv=1, one_pt=True):
        if derv == 1:
            return self.take_first_derivative(t, x, one_pt=one_pt)
        elif derv == 2:
            return self.take_second_derivative(t, x, one_pt=one_pt)
        elif derv == 3:
            return self.take_third_derivative(t, x, one_pt=one_pt)
        elif derv == 4:
            return self.take_forth_derivative(t, x, one_pt=one_pt)
        else:
            if one_pt:
                return x
            else:
                return x[1:-3]

    def build_ThetaX(self, t, X):
        t = np.array(t)
        self.t = t
        predictors = {}
        labels = {}
        #print(t.shape)
        if self.use_preprocessing:
            t, X = self.preprocesser(t, X)
            #print(t.shape)

        for i, x in enumerate(X):
            x = np.array(x)
            t, x = self.smooth(t,x)
            t_s = self.get_interpolated_t(t)
            self.t = np.array(t)
            #print(self.t.shape)

            if self.use_preprocessing:
                dervs = []
                for d in range(self.k_spline):
                    if self.config['derivative_method'] == 'spline':
                        dx = self.x_splines[i].interpolate(t_s, d)
                        _, dx = self.smooth(t_s, dx, coef=self.config['derivative_smooth_coef'] if d > 0 else None)
                        self.t = t_s
                    elif self.config['derivative_method'] == 'one_pt':
                        shift = (-1*(self.k_spline-1-d))
                        if shift == 0:
                            shift = None
                        
                        dx = self.take_derivative(t, x, d, one_pt=True)[:shift]
                        self.t = t[:-1*(self.k_spline-1)]
                    elif self.config['derivative_method'] == 'five_pt':
                        dx = self.take_derivative(t, x, d, one_pt=False)
                        self.t = t[1:-3]
                    else:
                        print('something went wrong!')
                        return
                    
                    dervs.append(dx)
                    if d == 0:
                        lab = 'x_%s'%i
                    else:
                        lab = 'd'+'x'*d+'_%s'%i
                    #print(self.t.shape, dx.shape)
                    # plt.plot(self.t, dx, '.--')
                    # plt.title(lab)
                    # plt.show()

                    if self.order_predict == d:
                        predictors[lab] = dx
                    else:
                        labels[lab] = dx
                

            else:
                dx = self.take_first_derivative(t, x)[:-2]
                dxx = self.take_second_derivative(t, x)[:-1]
                dxxx = self.take_third_derivative(t, x)[:]
                #dxxxx = self.take_forth_derivative(t, x)
                x = x[:-3]
                self.t = t[:-3]

                if self.order_predict == 0:
                    predictors['x_%s'%i] = x
                    labels['dx_%s'%i] = dx
                    labels['dxx_%s'%i] = dxx
                    labels['dxxx_%s'%i] = dxxx
                    #labels['dxxxx_%s'%i] = dxxxx
                elif self.order_predict == 1:
                    labels['x_%s'%i] = x
                    predictors['dx_%s'%i] = dx
                    labels['dxx_%s'%i] = dxx
                    labels['dxxx_%s'%i] = dxxx
                    #labels['dxxxx_%s'%i] = dxxxx
                elif self.order_predict == 2:
                    labels['x_%s'%i] = x
                    labels['dx_%s'%i] = dx
                    predictors['dxx_%s'%i] = dxx
                    labels['dxxx_%s'%i] = dxxx
                    #labels['dxxxx_%s'%i] = dxxxx
                elif self.order_predict == 3:
                    labels['x_%s'%i] = x
                    labels['dx_%s'%i] = dx
                    labels['dxx_%s'%i] = dxx
                    predictors['dxxx_%s'%i] = dxxx
                    #labels['dxxxx_%s'%i] = dxxxx
                elif self.order_predict == 4:
                    labels['x_%s'%i] = x
                    labels['dx_%s'%i] = dx
                    labels['dxx_%s'%i] = dxx
                    labels['dxxx_%s'%i] = dxxx
                    #predictors['dxxxx_%s'%i] = dxxxx
                else:
                    print('order_predict must be 0, 1, or 2.')
                    return

        if self.include_t:
            labels['t'] = t

        if self.include_poly_predictor:
            labels = {**labels, **predictors}
        
        orig_labels = list(labels.keys())
        for l in orig_labels:
            for i in range(2,self.poly_order+1):
                labels['%s^%s'%(l,i)] = labels[l]**i


        if self.include_poly_predictor:
            labels = {l:v for l, v in labels.items() if l not in predictors.keys()}

        self.ThetaX = np.array(list(labels.values()))
        self.labs = np.array(list(labels.keys()))
        self.b = np.array(list(predictors.values()))
        self.b_labs = np.array(list(predictors.keys()))
        
        #print(self.ThetaX.shape, self.labs.shape, self.b.shape, self.b_labs.shape)
        self.labels = labels
        self.predictors = predictors
        
        return (self.ThetaX, self.labs), (self.b, self.b_labs)

    def STLS(self, df_X, b, optimizer, opti_vars=None, iterates=3):
        coeffs_all = {col:0 for col in df_X.columns}
        for i in range(iterates):
            if opti_vars:
                coeffs = optimizer(df_X, b, **opti_vars)
            else:
                coeffs = optimizer(df_X, b)
            remain = []
            
            percents = np.abs(coeffs)/np.sum(np.abs(coeffs))
            i = 0
            for col, beta in zip(df_X.columns, coeffs):
                if np.abs(beta) < self.thresh or percents[i] < self.config['percent_thresh']:
                    beta = 0
                else:
                    remain.append(col)
                coeffs_all[col] = beta
                i += 1
            df_X = df_X[remain]
        return coeffs_all

    # def lasso(self, df_X, b):
    #     return
    

    def ridge(self, df_X, b):
        X = np.asmatrix(df_X.to_numpy())
        b = np.array(b)
        A = np.matmul(X.T,X)
        I = np.identity(A.shape[0])
        start = time.time()
        r = Lasso(self.alpha).fit(X,b)
        x = r.coef_
        #x = np.linalg.solve(2*(A+self.alpha*I), np.matmul(X.T,b).T)#1/2*np.matmul(np.matmul(np.linalg.inv(A+self.alpha*I),X.T), b)
        print(x.shape)
        end = time.time()
        if self.use_mlflow:
            mlflow.log_metric('optimizer_time (sec)', end-start)
            mlflow.log_param('ridge/alpha', self.alpha)
        return x

    def ridge_solve(self, df_X, b):
        X = np.asmatrix(df_X.to_numpy())
        b = np.array(b)
        A = np.matmul(X.T,X)
        I = np.identity(A.shape[0])
        start = time.time()
        x = np.linalg.solve((A+self.alpha*I), np.matmul(X.T,b).T)#1/2*np.matmul(np.matmul(np.linalg.inv(A+self.alpha*I),X.T), b)
        print(x.shape)
        end = time.time()
        if self.use_mlflow:
            mlflow.log_metric('optimizer_time (sec)', end-start)
            mlflow.log_param('ridge/alpha', self.alpha)
        return x

    def ridge_gm(self, df_X, b, alpha=False, tol=1e-6):
        if self.use_mlflow:
            mlflow.log_param('ridge_gm/alpha', self.alpha)
            mlflow.log_param('ridge_gm/tol', tol)
        X = np.asmatrix(df_X.to_numpy())
        b = np.array(b)
        A = np.matmul(X.T,X)
        I = np.identity(A.shape[0])
        print(X.shape, b.shape, A.shape, A.shape, I.shape)
        start = time.time()
        x, results = gradient_method((A+self.alpha*I), np.matmul(X.T,b).T, np.asmatrix(np.zeros(len(A))+1).T, alpha=False, tol=tol)
        end = time.time()
        if self.use_mlflow:
            mlflow.log_metric('optimizer_time (sec)', end-start)
            mlflow.log_dict(results, 'ridge_gm/results.json')
        self.mlflow_save_results(results)
        return x

    def ridge_cgm(self, df_X, b, tol=1e-6):
        if self.use_mlflow:
            mlflow.log_param('ridge_cgm/alpha', self.alpha)
            mlflow.log_param('ridge_cgm/tol', tol)
        X = np.asmatrix(df_X.to_numpy())
        b = np.array(b)
        A = np.matmul(X.T,X)
        I = np.identity(A.shape[0])
        start = time.time()
        x, results = conj_gradient_method((A+self.alpha*I), np.matmul(X.T,b).T, np.asmatrix(np.zeros(len(A))+1).T, tol=tol)
        end = time.time()
        if self.use_mlflow:
            mlflow.log_metric('optimizer_time (sec)', end-start)
            mlflow.log_dict(results, 'ridge_cgm/results.json')
        self.mlflow_save_results(results)
        return x

    def ridge_jor(self, df_X, b, omega=1, tol=1e-6):
        if self.use_mlflow:
            mlflow.log_param('ridge_jor/alpha', self.alpha)
            mlflow.log_param('ridge_jor/tol', tol)
            mlflow.log_param('ridge_jor/omega', omega)
        X = np.asmatrix(df_X.to_numpy())
        b = np.array(b)
        A = np.matmul(X.T,X)
        I = np.identity(A.shape[0])
        start = time.time()
        x, results = JOR((A+self.alpha*I), np.matmul(X.T,b).T, np.asmatrix(np.zeros(len(A))).T, omega=omega, tol=tol)
        end = time.time()
        if self.use_mlflow:
            mlflow.log_metric('optimizer_time (sec)', end-start)
            mlflow.log_dict(results, 'ridge_jor/results.json')
            self.mlflow_save_results(results)
        return x

    def ridge_gem(self, df_X, b):
        if self.use_mlflow:
            mlflow.log_param('ridge_gem/alpha', self.alpha)
        X = np.asmatrix(df_X.to_numpy())
        b = np.array(b)
        A = np.matmul(X.T,X)
        I = np.identity(A.shape[0])
        start = time.time()
        x = GEM((A+self.alpha*I), np.matmul(X.T,b).T)
        end = time.time()
        if self.use_mlflow:
            mlflow.log_metric('optimizer_time (sec)', end-start)
        return x

    def mlflow_save_results(self, results):
        for x, err, i in zip(results['x_k'], results['error'], results['iterations']):
            mlflow.log_metrics({f'beta/{k}':v for k, v in zip(self.labs, x)}, step=i)
            mlflow.log_metric('error', err, step=i)

    # def get_error(self, A,b,x):
    #     return np.linalg.norm(b-A*x, 2)

    # def gradient_method_iteration(self, A,b,x_0, alpha=None):
    #     r_k = b-A*x_0
    #     if alpha is None:
    #         alpha = np.dot(r_k.T, r_k)/np.dot(r_k.T, A*r_k)
    #     return x_0 + alpha[0,0]*r_k

    # def gradient_method(self, A,b,x_0, tol=1e-6, alpha=None):
    #     results = {'x_k' : [x_0], 'error' : [self.get_error(A,b,x_0)], 'iterations' : 0}
    #     #mlflow.log_metrics({f'beta/{k}':v for k, v in zip(self.labs, results['x_k'][-1])}, step=results['iterations'][-1])
    #     #mlflow.log_metric('error', results['error'][-1], step=results['iterations'][-1])
    #     while(results['error'][-1] > tol):
    #         results['x_k'].append(self.gradient_method_iteration(A,b,results['x_k'][-1], alpha=alpha))
    #         results['error'].append(self.get_error(A,b,results['x_k'][-1]))
    #         results['iterations'] += 1
    #         #mlflow.log_metrics({k:v for k, v in zip(self.labs, results['x_k'][-1])}, step=results['iterations'][-1])
    #         #mlflow.log_metric('error', results['error'][-1], step=results['iterations'][-1])
    #     return results['x_k'][-1], results
                
    # def conj_gradient_method(self, A,b,x_0, tol=1e-6, alpha=None):
    #     results = {'x_k' : [x_0], 'error' : [self.get_error(A,b,x_0)], 'iterations' : 0}
    #     p = b-A*x_0
    #     while(results['error'][-1] > tol):
    #         r = b-A*results['x_k'][-1]
    #         alpha = np.dot(p.T, r)/np.dot(p.T, A*p)
    #         r_1 = r - alpha[0,0]*A*p
    #         beta_k = np.dot((A*p).T, r_1)/np.dot((A*p).T, p)
    #         #print(beta_k)
            
    #         x_1 = results['x_k'][-1] + alpha[0,0]*p
    #         p = r_1 - beta_k[0,0]*p
            
    #         results['x_k'].append(x_1)
    #         results['error'].append(self.get_error(A,b,x_1))
    #         results['iterations'] += 1
    #         #p = get_next_dir_p(A,b,x_0,p)
    #         #print(results['error'][-1])
    #     return results['x_k'][-1], results

    # def get_D(self, A):
    #     D = A*1
    #     for i, a in enumerate(A):
    #         for j, x in enumerate(a.T):
    #             if i != j:
    #                 D[i,j] = 0
    #     #print('D:', D)
    #     return D

    # def JOR_iteration(self, A,b,x_0, omega=1):
    #     return x_0 + omega*np.linalg.inv(self.get_D(A))*(b - A*x_0)

    # def JOR(self, A,b,x_0,omega=1,tol=1e-6):
    #     results = {'x_k' : [x_0], 'error' : [self.get_error(A,b,x_0)], 'iterations' : 0}
    #     while(results['error'][-1] > tol):
    #         results['x_k'].append(self.JOR_iteration(A,b,results['x_k'][-1],omega))
    #         results['error'].append(self.get_error(A,b,results['x_k'][-1]))
    #         results['iterations'] += 1
    #     return results['x_k'][-1], results
            
    def fit(self, t, X, optimizer=None, opti_vars=None, iterates=3):
        if optimizer is None:
            optimizer = self.ridge


        X_vars, bs = self.build_ThetaX(t, X)


        df_X = pd.DataFrame(X_vars[0].T, columns=X_vars[1])
        df_X.index = self.t
        self.df_X = df_X
        coefs_all = {k: None for k in bs[1]}
        non_zero_coefs_all = {k: None for k in bs[1]}
        for b, b_lab in zip(bs[0], bs[1]):
            print(b_lab)
            coefs = self.STLS(df_X, b, optimizer, opti_vars=opti_vars, iterates=iterates)
            coefs_all[b_lab] = coefs#{k:coefs[k] for k in bs[1]}
            #print(coefs)
            non_zero_coefs_all[b_lab] = {k:coefs[k] for k in coefs.keys() if np.abs(coefs[k]) > self.thresh}

        self.coefs = coefs_all
        self.non_zero_coefs = non_zero_coefs_all

        return self.coefs

    def predict(self, t, X):
        X_vars, bs = self.build_ThetaX(t, X)
        return {k:np.matrix(X_vars[0])*np.array(self.coefs[k].values()) for k in self.coefs.keys()}


class K_Poly_Spline():

    def __init__(self, x, y, k=3, method=None, config=None) -> None:
        self.k = k
        self.x = x
        self.y = y
        self.len_x = len(x)
        self.method = method
        self.config = config

        self.b = np.zeros((k+1)*(len(y)-1))
        self.len_b = len(self.b)
        #self.b[:len(y)] = y

        self.A = np.zeros((len(self.b), len(self.b)))
        
        self.build_A_and_b()
        print('\t\tbuilt A and b. SIZE:', self.A.shape)
        #print(self.A)

        self.f = self.build_f_coefs()
        #print('found f coefs')

    def get_coef_values(self, x_i, derivative=0):
        coefs = np.zeros(self.k+1)
        for i, c in enumerate(coefs[:len(coefs)-derivative]):
            if i == self.k-derivative:
                c = math.factorial(self.k-i)/math.factorial(self.k-i-derivative)
            else:
                c = math.factorial(self.k-i)/math.factorial(self.k-i-derivative)*x_i**(self.k-i-derivative)
            coefs[i] = c
        return coefs

    def position_functions(self):
        matrix_i = 0
        matrix_j = 0
        for i, y_i in enumerate(self.y[:-1]):
            cs_1 = self.get_coef_values(self.x[i], derivative=0)
            cs_2 = self.get_coef_values(self.x[i+1], derivative=0)
            self.b[matrix_i:matrix_i+2] = [y_i, self.y[i+1]]
            for c in [cs_1, cs_2]:
                self.A[matrix_i, matrix_j:matrix_j+self.k+1] = c
                matrix_i += 1
            matrix_j += self.k+1
        return matrix_i

    def derivative_functions(self, matrix_i):
        for d in range(1,self.k):
            matrix_j = 0
            for i in range(1,self.len_x-1):
                cs = self.get_coef_values(self.x[i], d)
                cs_inv = -1*cs
                self.A[matrix_i, matrix_j:matrix_j+len(cs)*2] = list(cs)+list(cs_inv)
                matrix_i += 1
                matrix_j += len(cs)
        return matrix_i

    # def edge_functions(self, matrix_i):
    #     num_dervs_needed = self.k-2
    #     for d in range(self.k-1, self.k-1-num_dervs_needed, -1):
    #         cs_0 = self.get_coef_values(self.x[0], d)
    #         cs_end = self.get_coef_values(self.x[-1], d)

    #         self.A[matrix_i, 0:self.k+1] = cs_0
    #         self.A[matrix_i+1, self.len_b-len(cs_0):self.len_b] = cs_end
    #     return matrix_i

    def edge_functions(self, matrix_i):
        num_dervs_needed = self.k//2
        c = 0
        for d in range(self.k-1, self.k-num_dervs_needed-1, -1):
            #print(d)
            cs_0 = self.get_coef_values(self.x[0], d)
            cs_end = self.get_coef_values(self.x[-1], d)

            self.A[matrix_i, 0:self.k+1] = cs_0
            c+=1
            matrix_i +=1
            if c < self.k-1:
                self.A[matrix_i, self.len_b-len(cs_0):self.len_b] = cs_end
                c+=1
                matrix_i +=1
        return matrix_i


    def build_A_and_b(self):
        i = self.position_functions()
        #print(i)
        i = self.derivative_functions(i)
        #print(i)
        i = self.edge_functions(i)
        return

    def solve_A_b(self):
        self.A = np.asmatrix(self.A)
        self.b = np.asmatrix(self.b).T
        b = np.matmul(self.A.T, self.b)
        A = np.matmul(self.A.T, self.A)
        if self.config:
            config = self.config
        else:
            config = {'ridge_gm' : {'tol' : 1e-5}, 'ridge_cgm' : {'tol' : 1e-5}, 'ridge_jor' : {'tol' : 1e-5}}
        if self.method == 'GEM':
            return GEM(A, b)
        elif self.method == 'GM':
            x, results = gradient_method(A, b, np.zeros(len(b))+1, **config['ridge_gm'])
            return x
        elif self.method == 'CGM':
            x, results = conj_gradient_method(A, b, np.zeros(len(b))+1, **config['ridge_cgm'])
        elif self.method == 'rref':
            return rref(A, b)
        elif self.method == 'JOR':
            x, results = JOR(A, b, np.zeros(len(b))+1, **config['ridge_jor'])
            return x
        elif self.method == 'inv':
            return np.matmul(np.linalg.inv(A), b)
        else:
            return np.linalg.solve(A, b)

    def build_f_coefs(self):
        #print('solving A b...')
        coefs = self.solve_A_b()
        #print('solved.')
        coef_dict = {}
        i = 0
        for x in self.x[1:]:
            coef_dict[x] = np.array(coefs[i:i+self.k+1])
            i += self.k+1

        return coef_dict

    def get_pred_y(self, x_s, derivative=0):
        keys = list(self.f.keys())
        #print(x_s)
        
        if x_s >= self.x[0] and x_s <= self.x[-1]:
            if x_s <= keys[0]:
                res = keys[0]
            else:
                for i, x in enumerate(keys[:-1]):
                    if x_s >= x and x_s <= keys[i+1]:
                        break
                #print(i, '\n')
                res = keys[i+1]
        else:
            print('x_s is out of predictable range, returning np.nan')
            return np.nan
        
        if derivative >= 0 and derivative < self.k:
            c = np.array(self.get_coef_values(x_s, derivative=derivative))
            # print(c)
            # print(self.f[res])
            # print()
            return np.sum(c*self.f[res])
        else:
            print('derivative too high or does not make sense. returning None')
            return np.nan

    def interpolate(self, list_x, derivative=0):
        return np.array([self.get_pred_y(x, derivative=derivative) for x in list_x])

import random

class E_SINDy(object):

    def __init__(self, config, n_models = 50,
                    point_ratio=0.1,
                    optimizer = 'ridge_gm',
                    random_seed = 42,
                    use_mlflow=True,
                    experiment_name='sindy_model', 
                    run_name='sindy_base') -> None:
        self.config = config
        self.n_models = n_models
        if point_ratio <= 0:
            print('point_ratio doesn\'t make sence.  Setting to 0.1')
            point_ratio = 0.1
        self.point_ratio = point_ratio
        self.optimizer = optimizer
        self.random_seed = random_seed
        self.use_mlflow = use_mlflow
        self.experiment_name = experiment_name
        self.run_name = run_name

        if use_mlflow:
            mlflow.log_param('n_model', n_models)
            mlflow.log_param('point_ratio', point_ratio)
            mlflow.log_param('optimizer', optimizer)
            mlflow.log_param('random_seed', random_seed)
        return

    def preprocesser(self, t, X, n=None):
        t_s = np.array(list(np.linspace(t[0], t[-1], int((t[-1] - t[0])//self.interpolated_dt))))
        #print(t_s[0], t_s[-1])
        for i, x in enumerate(X):
            start = time.time()
            spline = K_Poly_Spline(t, x, self.k_spline, self.spline_method, config=self.config)
            #print(list(spline.f.keys())[-1])
            end = start - time.time()
            X[i] = spline.interpolate(t_s)
            self.all_spline.append(spline)
            if self.use_mlflow:
                if n is None:
                    mlflow.log_artifact(f'preprocessing/spline_x_{i}', spline.f)
                else:
                    mlflow.log_artifact(f'preprocessing/spline_x_{i}_model_{n}', spline.f)
                mlflow.log_metric(f'preprocessing/x_{i}_solve_time', end)
        return t_s, X

    def run_base_sindy(self, t, X, poly_order=5,
                    include_t=False,
                    include_poly_predictor=False, 
                    order_predict=1, 
                    thresh=1e-6, 
                    alpha=1.0,
                    use_preprocessing = True,
                    k_spline = 3,
                    spline_method = None,
                    interpolated_dt = 0.1,
                    stls_iterates=3):

        sindy = BaseSINDy(config = self.config, poly_order=poly_order, 
                            include_t=include_t, 
                            include_poly_predictor=include_poly_predictor, 
                            order_predict=order_predict, 
                            thresh=thresh, 
                            alpha=alpha, 
                            use_preprocessing = use_preprocessing,
                            k_spline = k_spline,
                            spline_method = spline_method,
                            interpolated_dt = interpolated_dt,
                            use_mlflow=self.use_mlflow)

        if self.optimizer == 'ridge':
            optimizer = sindy.ridge
            opti_vars = None
        elif self.optimizer == 'ridge_gm':
            optimizer = sindy.ridge_gm
            opti_vars = self.config['ridge_gm']
        elif self.optimizer == 'ridge_cgm':
            optimizer = sindy.ridge_cgm
            opti_vars = self.config['ridge_cgm']
        elif self.optimizer == 'ridge_jor':
            optimizer = sindy.ridge_jor
            opti_vars = self.config['ridge_jor']
        elif self.optimizer == 'ridge_gem':
            optimizer = sindy.ridge_gem
            opti_vars = None
        else:
            optimizer = self.optimizer
            opti_vars = None
        
        coefs = sindy.fit(t, X, optimizer=optimizer, opti_vars=opti_vars, iterates=stls_iterates)
        self.all_spline.append(sindy.x_splines)
        return coefs, sindy.df_X

    def smooth(self, t, x, t_s=None):
        coef=self.config['smooth_coef']
        spl = UnivariateSpline(t, x)
        spl.set_smoothing_factor(coef)
        #print('smoothing:', coef)
        if t_s is None:
            return t, spl(t)
        else:
            return t_s, spl(t_s) 

    def preprocess(self, t, X):
        t_s = np.linspace(t[0], t[-1], 100)

        #print(t_temp[0], t_temp[-1])
        X_temp = []
        for x in X:
            _, x = self.smooth(t, x, t_s)
            X_temp.append(x)
        X_temp = np.array(X_temp)
        return t_s, X_temp

    def fit(self, t, X, 
                    poly_order=5,
                    include_t=False,
                    include_poly_predictor=False, 
                    order_predict=1, 
                    thresh=1e-6, 
                    alpha=1.0,
                    use_preprocessing = True,
                    k_spline = 3,
                    spline_method = None,
                    interpolated_dt = 0.1,
                    stls_iterates = 3):
        self.poly_order = poly_order
        self.include_t = include_t
        self.include_poly_predictor = include_poly_predictor
        self.thresh = thresh
        self.alpha = alpha
        self.use_preprocessing = use_preprocessing
        self.k_spline = k_spline
        self.spline_method = spline_method
        self.interpolated_dt = interpolated_dt
        self.stls_iterates = stls_iterates
        
        all_coefs = {}
        all_dfs = {}
        self.all_spline = []

        # if self.use_preprocessing:
        #     t, X = self.preprocess(t, X)
        self.t = t
        self.X = X
        for i in range(self.n_models):
            print('running iteration: ', i)
            #random.seed(self.random_seed)
            if self.point_ratio < 1:
                sample_index = random.sample(list(range(len(t))), int(len(t)*self.point_ratio))
                sample_index.sort()
            elif self.point_ratio > 1:
                sample_index = random.sample(list(range(len(t))), int(self.point_ratio))
                sample_index.sort()
            else:
                sample_index = list(range(len(t)))

            t_temp = np.array(t[sample_index])
            #print(t_temp[0], t_temp[-1])
            X_temp = []
            for x in X:
                X_temp.append(np.array(x[sample_index]))
            X_temp = np.array(X_temp)
            print('\t start sindy..')
            coefs, df_X = self.run_base_sindy(t_temp, X_temp, poly_order=poly_order, 
                                        include_t=include_t, 
                                        include_poly_predictor=include_poly_predictor, 
                                        order_predict=order_predict, 
                                        thresh=thresh, 
                                        alpha=alpha,
                                        use_preprocessing = use_preprocessing,
                                        k_spline = k_spline,
                                        spline_method = spline_method,
                                        interpolated_dt = interpolated_dt, 
                                        stls_iterates=stls_iterates)
            print('\t sindy end.')
            
            all_dfs[i] = df_X
            for k, v in coefs.items():
                if k not in all_coefs.keys():
                    all_coefs[k] = [v]
                else:
                    all_coefs[k].append(v)
                if self.use_mlflow:
                    mlflow.log_metric(f'coef_average_{k}', np.mean(all_coefs[k]))
                    mlflow.log_metric(f'coef_standard_dev_{k}', np.std(all_coefs[k]))
        #print(all_coefs)
        self.coefs =  {k : {'mean' : pd.DataFrame(v).mean().to_dict(), 'std' : pd.DataFrame(v).std().to_dict(), 'df' : pd.DataFrame(v)} for k, v in all_coefs.items()}
        self.all_dfs = all_dfs
        if self.use_mlflow:
            mlflow.log_artifact(self.coefs, 'coefs.json')
        return self.coefs

    def predict(self):
        return
        

def GEM_step(A, b):
    operations = 0
    A_temp = A.copy()
    b_temp = b.copy()
    for i in range(1, A.shape[0]):
        #print(A)
        for j in range(0, A.shape[1]):
            A_temp[i,j] = A[i,j] - A[i,0]/A[0,0] * A[0,j]
            operations +=1
        b_temp[i] = b[i] - A[i,0]/A[0,0] * b[0]
        operations +=1
    return A_temp, b_temp, operations

def back_substitute(A,b):
    n = len(b)
    x = np.zeros(n)

    for i in range(n-1, -1, -1):
        temp = b[i]
        for j in range(n-1, i, -1):
            temp -= x[j]*A[i,j]

        x[i] = temp/A[i,i]
    return x

def GEM(A, b):
    o = 0
    #print(A, b)
    #print(A.shape, b.shape)
    A = np.matrix(A)
    #print('1\t', A)
    b = np.array(b)
    #print(A.shape, b.shape)
    for i in range(A.shape[0]-1):
        A_step, b_step, operations = GEM_step(A[i:,i:], b[i:])
        A[i:,i:] = A_step
        #print(i, '\t', A)
        b[i:] = b_step
        o += operations
        #print(i, A, b)
    x = back_substitute(A,b)
    return x

def get_error(A,b,x):
    return np.linalg.norm(b-np.matmul(A,x), 2)

def gradient_method_iteration(A,b,x_0, tol=1e-6, alpha=False):
    r_k = (b-np.matmul(A,x_0))
    #print(r_k.shape, A.shape, np.matmul(A,x_0).shape, x_0.shape)
    #print(r_k)
    if not alpha:
        #print(np.dot(r_k.T, A*r_k))
        alpha = np.dot(r_k.T, r_k)/np.dot(r_k.T, np.matmul(A,r_k))
    if isinstance(alpha, float):
        return x_0 + alpha*r_k
    else:
        return x_0 + alpha[0,0]*r_k

def gradient_method(A,b,x_0, tol=1e-6, alpha=False):
    results = {'x_k' : [x_0], 'error' : [get_error(A,b,x_0)], 'iterations' : 0}
    while(results['error'][-1] > tol):
        results['x_k'].append(gradient_method_iteration(A,b,results['x_k'][-1],tol,alpha))
        results['error'].append(get_error(A,b,results['x_k'][-1]))
        results['iterations'] += 1

        if results['iterations']%100000 == 0:
            print('run away!', results['iterations'])
            print(results['error'][0], results['error'][-1])
    return results['x_k'][-1], results

def rref(A, b):
    #print(A, b)
    c = []
    for a, b in zip(A, b):
        c.append(list(a)+[b])
    m, i = sympy.Matrix(c).rref()
    return np.array(m[:, -1]).reshape(len(np.array(m)))

def get_D(A):
        D = np.array(A)
        for i, a in enumerate(A):
            for j, x in enumerate(a.T):
                if i != j:
                    D[i,j] = 0
        #print('D:', D)
        return D

def JOR_iteration(A,b,x_0, omega=1):
    return x_0 + np.matmul(omega*np.linalg.inv(get_D(A)),(b - np.matmul(A,x_0)))

def JOR(A,b,x_0,omega=1,tol=1e-6):
    results = {'x_k' : [x_0], 'error' : [get_error(A,b,x_0)], 'iterations' : 0}
    while(results['error'][-1] > tol):
        results['x_k'].append(JOR_iteration(A,b,results['x_k'][-1],omega))
        results['error'].append(get_error(A,b,results['x_k'][-1]))
        results['iterations'] += 1

        if results['iterations']%100000 == 0:
            print('run away!', results['iterations'])
            print(results['error'][0], results['error'][-1])
    return results['x_k'][-1], results

def conj_gradient_method(A,b,x_0, tol=1e-6):
    b = np.asmatrix(b).T
    x_0 = np.asmatrix(x_0).T
    A = np.asmatrix(A)
    print(b.shape, A.shape, x_0.shape)
    results = {'x_k' : [x_0], 'error' : [get_error(A,b,x_0)], 'iterations' : 0}
    
    p = b-np.matmul(A,x_0)
    print(p.shape, 'p')
    while(results['error'][-1] > tol):
        r = b-np.matmul(A, results['x_k'][-1])
        #print(p.T.shape, np.matmul(A,p).shape)
        alpha = np.dot(p.T, r)/np.dot(p.T, np.matmul(A,p))
        r_1 = r - alpha[0,0]*np.matmul(A,p)
        beta_k = np.dot(np.matmul(A,p).T, r_1)/np.dot(np.matmul(A,p).T, p)
        #print(beta_k)
        
        x_1 = results['x_k'][-1] + alpha[0,0]*p
        p = r_1 - beta_k[0,0]*p
        
        results['x_k'].append(x_1)
        results['error'].append(get_error(A,b,x_1))
        results['iterations'] += 1
        #p = get_next_dir_p(A,b,x_0,p)
        #print(results['error'][-1])
        if results['iterations']%100000 == 0:
            print('run away!', results['iterations'])
            print(results['error'][0], results['error'][-1])
            
    return results['x_k'][-1], results
