import numpy as np
import matplotlib.pyplot as plt


def simple_statistic(i,j,n,p,Y):
    return np.sum((Y[i,:] - Y[j,:])**2 - 2) / (n**(p-1))

def two_hop_statistic(i,j,n,p,Y, Omega):
    print('calculating statistic')
    stat = 0
    num_entries = 0
    for l in range(n):
        if l != i and l != j:
            it_one = np.nditer(Y[l, :], flags=['multi_index'])
            while not it_one.finished:
                index_one = (l,) + it_one.multi_index
                it_two = np.nditer(Y[l, :], flags=['multi_index'])
                while not it_two.finished:
                    index_two = (l,) + it_one.multi_index
                    if Omega[index_one] == 1 and Omega[index_two] == 0:
                        stat += Y[index_one] * Y[index_two] * Y[(i,) + it_one.multi_index] * Y[(j,) + it_two.multi_index]
                        num_entries += 1
                    it_two.iternext()
                    it_one.iternext()
    return stat / num_entries

p = 3
prob = .4
n = 3
lam = 5

size_list = []
for _ in range(p):
    size_list.append(n)


x = 2*(np.random.binomial(1, prob, n)-.5)
# print(x)
Y = np.tensordot(x,x, axes=0)
for k in range(p-2):
    Y = np.tensordot(Y, x, axes=0)
Y = lam*Y + np.random.normal(0,1,size_list)
print(Y)

Omega = np.random.binomial(1, .5, size_list)
print(Omega)

two_hop_statistic(0,1,n,p,Y,Omega)
