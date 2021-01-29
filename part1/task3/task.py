import numpy as np
from matplotlib import pyplot as plt

EPOCH = 100
learning_rate = 0.1
tolerance = 1e-7

# calculate gradient using finite difference
def finite_difference_gradient(FX, x):  
    dfx = np.copy(x)
    # calculate gradient on x(different directions) one by one
    for i in range(len(x)):
        x_plus = np.copy(x)
        x_minus = np.copy(x)
        x_plus[i] += 1e-5
        x_minus[i] -= 1e-5
        dfx[i] = (FX(x_plus) - FX(x_minus)) / (2*1e-5)
    return dfx

# graduent descent on FX, optimize x
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

# calculate polynomial according to a
def quadratic_polynomial(a, x):   
    return a[0] + a[1]*x[0]**2 + a[2]*x[1]**2 + a[3]*x[2]**2 + a[4]*x[0]*x[1] + \
           a[5]*x[0]*x[2] + a[6]*x[1]*x[2] + a[7]*x[0] + a[8]*x[1] + a[9]*x[2]
 
# add some control on a to convert objective function into format of (A+Bx1+Cx2+Dx3)^2
# which is definitely convex function          
def a_cal(convex_paras): 
    a = np.zeros(10)
    # a0=A^2, a1=A^2, a2=A^2, a3=A^2
    # a4=2BC, a5=2BD, a6=2CD, a7=2AB a8=2AC, a9=2AD
    a[0:4] = convex_paras[0:4]**2
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
    # calculate a from convex_paras
    a = a_cal(convex_paras)
    # minimum when A+Bx1+Cx2+Dx3 = 0
    # initial estimate: x0 = 0s, where f(x0) = A^2 is in range(0,4)
    x = np.zeros(3)
    FX = lambda x: quadratic_polynomial(a, x)
    # solve minimum by using gradient descent
    fx_list, solution = gradient_descent(FX, x)
    
    # plot gradient descent process and show results
    plt.plot(np.arange(len(fx_list)), fx_list)
    plt.xlabel("epoch", fontsize=15)
    plt.ylabel("fx", fontsize=15)
    plt.text(len(fx_list)/2, (fx_list[len(fx_list)-1]+fx_list[0])/10, \
             "Epoch: %d\nminimum: %.4f\nsolution: %.4f %.4f %.4f" %(len(fx_list), np.round(fx_list[len(fx_list)-1]), \
               solution[0], solution[1], solution[2]), fontsize=12)
    plt.show()

