import numpy as np
from matplotlib import pyplot as plt

EPOCH = 100
learning_rate = 0.1
tolerance = 1e-7


def finite_difference_gradient(FX, x):  
    dfx = np.copy(x)
    for i in range(len(x)):
        x_plus = np.copy(x)
        x_minus = np.copy(x)
        x_plus[i] += 1e-5
        x_minus[i] -= 1e-5
        dfx[i] = (FX(x_plus) - FX(x_minus)) / (2*1e-5)
    
    return dfx

def gradient_descent(FX, x):    
    fx = FX(x)
    fx_list = [fx]
    for n in range(EPOCH):
        fx_last = fx
        dfx = finite_difference_gradient(FX, x) 
        x += -learning_rate * dfx  
        fx = FX(x)

        if fx - fx_last > -tolerance:
            break
        fx_list.append(fx)     
    return fx_list, x

#(A+Bx1+Cx2+Dx3)^2
def quadratic_polynomial(a, x):   
    return a[0] + a[1]*x[0]**2 + a[2]*x[1]**2 + a[3]*x[2]**2 + a[4]*x[0]*x[1] + \
           a[5]*x[0]*x[2] + a[6]*x[1]*x[2] + a[7]*x[0] + a[8]*x[1] + a[9]*x[2]
           
def a_cal(convex_paras): 
    a = np.zeros(10)
    a[0:4] = convex_paras[0:4]*convex_paras[0:4]
    a[4] = 2*convex_paras[1]*convex_paras[2]
    a[5] = 2*convex_paras[1]*convex_paras[3]
    a[6] = 2*convex_paras[2]*convex_paras[3]
    a[7] = 2*convex_paras[0]*convex_paras[1]
    a[8] = 2*convex_paras[0]*convex_paras[2]
    a[9] = 2*convex_paras[0]*convex_paras[3]
    return a
           
if __name__ == '__main__':
    # convex_paras(A B C D) in range(-2,2)
    convex_paras = (np.random.rand(4)-0.5)*4
    a = a_cal(convex_paras)
    # A+Bx1+Cx2+Dx3 = 0
    # initial estimate: x = 0s, where f(x) = D^2 in range(0,4)
    x = np.zeros(3)
    FX = lambda x: quadratic_polynomial(a, x)
    fx_list, solution = gradient_descent(FX, x)
    plt.plot(np.arange(len(fx_list)), fx_list)
    plt.xlabel("epoch", fontsize=15)
    plt.ylabel("fx", fontsize=15)
    plt.text(len(fx_list)/2, (fx_list[len(fx_list)-1]+fx_list[0])/10, \
             "Epoch: %d\nminimum: %.4f\nsolution: %.4f %.4f %.4f" %(len(fx_list), np.round(fx_list[len(fx_list)-1]), \
               solution[0], solution[1], solution[2]), fontsize=12)
    plt.show()

