import numpy as np
from cvxopt import solvers, matrix
from numpy.linalg import norm, pinv
import scipy.optimize
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Reading the text files and converting to arrays 

with open('b_test_1.txt') as f1:
    b_list = [float(x) for x in f1.read().split()]

b = np.array(b_list)      #convert to array
print(type(b))
print(b)

with open('A_test_1.txt') as f2:
    A_list = []
    for line in f2:  # read rest of the lines
        A_list.append([float(x) for x in line.split()])

A = np.array(A_list)   #convert to array
print(type(A))
print(A)

with open('Q_test_1.txt') as f3:
    Q_list = []
    for line in f3:  # read rest of the lines
        Q_list.append([float(x) for x in line.split()])

Q = np.array(Q_list)   #convert to array
print(type(Q))
print(Q)

# dim = dimensions of A
# dim = (m, n)
m, n = np.shape(A)
print(m, n)


# Obtaining the optimal solution x*

# standard form of a quadratic program (CVXOPT)

# 1/2 P = Q, q_T = 0
P = 2 * Q

# print(P.shape)
# print(type(P))

q = np.zeros(n)

#required
P_m = matrix(P)
q_m = matrix(q)

#optional
A_m = matrix(A)
b_m = matrix(b)
G = np.zeros((n, n))
h = np.zeros(n)
G_m = matrix(G)
h_m = matrix(h)

sol = solvers.qp(P=P_m, q=q_m, G=G_m, h=h_m, A=A_m, b=b_m)['x']
sol = np.asarray(sol)
sol = sol.reshape(sol.shape[0])
print(sol)


x = np.ones((n))
x_T = np.transpose(x)
# print(x_T)
k = 0

epsilon = 10 ** (-6)   #constant stopping criterion
c_k = 1

L = np.ones((m))  # lambda
L_T = np.transpose(L)


def get_x_k(Q, A, b, c_k, L):
    A_T = np.transpose(A)
    f_x = 2 * Q + (c_k * A_T.dot(A))
    h_x = (c_k * A_T.dot(b))
    new_h_x = h_x - A_T.dot(L)
    x = pinv(f_x).dot(new_h_x)
    return x.reshape(x.shape[0])


def output(A, x_T, k, L, sol):

    # x = np.ones([n, 1])
    # x = np.ones((n))
    error_list = []
    c_k = 1
    while True:
        x = get_x_k(Q, A, b, c_k, L)
        if (norm((A.dot(x) - b)) <= epsilon):       #stopping condition
            print(k)
            return x, error_list
        error_list.append(get_error(x, sol))
        hx = A.dot(x) - b
        new_hx = hx.reshape((m))
        L = L + c_k * new_hx
        print(L)
        k = k + 1
        c_k = k * k

#relative error 

def get_error(x_k, sol):
    return np.linalg.norm(x_k - sol) / np.linalg.norm(sol)

x_k, error_list = output(A, x_T, k, L, sol)

print(get_error(x_k, sol))

errorlen = len(error_list)
plt.plot(np.arange(errorlen), error_list)
plt.title('Error Plot')
plt.show()
