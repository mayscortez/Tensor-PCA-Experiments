import numpy as np
import matplotlib.pyplot as plt
import time

np.random.seed(4)

def simple_statistic(i,j,n,p,Y):
    return np.sum((Y[i,:] - Y[j,:])**2 - 2) / (n**(p-1))

def two_hop_statistic(i,j,n,p,Y, Omega, x):
    # print('Calculating the statistic for i: ' + str(i) + ' and j: ' + str(j))

    stat = 0
    num_entries = 0
    for l in range(n):
        if l != i and l != j:

            # print('Using the index l : ' + str(l))

            it_one = np.nditer(Y[l, :], flags=['multi_index'])
            while not it_one.finished:
                index_one = (l,) + it_one.multi_index
                it_two = np.nditer(Y[l, :], flags=['multi_index'])
                while not it_two.finished:
                    index_two = (l,) + it_two.multi_index
                    # print('index one: ' + str(index_one))
                    # print('index two: ' + str(index_two))
                    if Omega[index_one] == 1 and Omega[index_two] == 0:
                        stat += Y[index_one] * Y[index_two] * Y[(i,) + it_one.multi_index] * Y[(j,) + it_two.multi_index]
                        num_entries += 1
                    it_two.iternext()
                it_one.iternext()
    return stat / num_entries

def two_hop_statistic_new(i,j,n,p,Y, Omega, x):
    # print('Calculating the statistic for i: ' + str(i) + ' and j: ' + str(j))

    stat = 0
    num_entries = 0
    for l in range(n):
        if l != i and l != j:
            # print('Using the index l : ' + str(l))
            it_one = np.nditer(Y[l, :], flags=['multi_index'])
            while not it_one.finished:
                index_one = (l,) + it_one.multi_index
                it_two = np.nditer(Y[l, :], flags=['multi_index'])
                while not it_two.finished:
                    index_two = (l,) + it_two.multi_index
                    # print('index one: ' + str(index_one))
                    # print('index two: ' + str(index_two))
                    if Omega[it_one.multi_index] == 1 and Omega[it_two.multi_index] == 0:
                        stat += Y[index_one] * Y[index_two] * Y[(i,) + it_one.multi_index] * Y[(j,) + it_two.multi_index]
                        num_entries += 1
                    it_two.iternext()
                it_one.iternext()
    return stat / num_entries

p = 3
num_iters = 1
# lam_grid = [.05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
lam_grid = [.025, .1, .5]
prob = .5


n = 25
error_25_one = np.zeros(len(lam_grid))
error_25_two = np.zeros(len(lam_grid))
size_list = []
for _ in range(p):
    size_list.append(n)
# print(size_list)


sign = 0
Omega = np.zeros(size_list[1:])
it = np.nditer(Omega, flags=['multi_index'])
while not it.finished:
    index = it.multi_index
    Omega[index] = sign
    sign = 1 - sign
    it.iternext()

index = 0
for lam in lam_grid:
    for i in range(num_iters):
        print(lam, i)
        x = 2*(np.random.binomial(1, prob, n)-.5)
        # print(x)
        Y = np.tensordot(x,x, axes=0)
        for k in range(p-2):
            Y = np.tensordot(Y, x, axes=0)
        Y = lam*Y + np.random.normal(0,1,size_list)
        # Y = lam*Y

        # print(Y)
        xhat_one = np.zeros((n))
        xhat_two = np.zeros((n))
        # Just set the sign of the first entry to the true one as we only care about signs
        xhat_one[0] = x[0]
        xhat_two[0] = x[0]


        for j in range(n):
            if j != 0:
                # Compute the Z statistic
                # print('Correct answer: ' + str(x[j]))
                Z = simple_statistic(0,j,n,p,Y)
                # print('Z statistic is: ' + str(Z))
                if Z <= 2*lam**2:
                    xhat_one[j] = xhat_one[0]
                else:
                    xhat_one[j] = (-1)*xhat_one[0]
                # print('Final answer: ' + str(xhat[j]))
                # print('new bit#########')
                # print(xhat_two[0])
                # print('Correct answer: ' + str(x[j]))
                Z = two_hop_statistic_new(0,j,n,p,Y,Omega, x)
                # print('Z statistic is: ' + str(Z))
                if Z >= 0:
                    xhat_two[j] = xhat_two[0]
                else:
                    xhat_two[j] = (-1)*xhat_two[0]
                # print('Final answer: ' + str(xhat_two[j]))
                # if xhat_two[j] != x[j]:
                #     print('WRONG DECISION######################################')
                # print('################')
        loss_one = np.abs(np.dot(xhat_one, x)) / n
        loss_two = np.abs(np.dot(xhat_two, x)) / n
        # print(loss_two)
        error_25_one[index] += loss_one / num_iters
        error_25_two[index] += loss_two / num_iters
    index += 1

#
# n = 50
# error_50_one = np.zeros(len(lam_grid))
# error_50_two = np.zeros(len(lam_grid))
# size_list = []
# for _ in range(p):
#     size_list.append(n)
# # print(size_list)
#
#
# sign = 0
# Omega = np.zeros(size_list)
# it = np.nditer(Omega, flags=['multi_index'])
# while not it.finished:
#     index = it.multi_index
#     Omega[index] = sign
#     sign = 1 - sign
#     it.iternext()
#
# index = 0
# for lam in lam_grid:
#     for i in range(num_iters):
#         x = 2*(np.random.binomial(1, prob, n)-.5)
#         # print(x)
#         Y = np.tensordot(x,x, axes=0)
#         for k in range(p-2):
#             Y = np.tensordot(Y, x, axes=0)
#         Y = lam*Y + np.random.normal(0,1,size_list)
#         # Y = lam*Y
#
#         # print(Y)
#         xhat_one = np.zeros((n))
#         xhat_two = np.zeros((n))
#         # Just set the sign of the first entry to the true one as we only care about signs
#         xhat_one[0] = x[0]
#         xhat_two[0] = x[0]
#
#
#         for j in range(n):
#             if j != 0:
#                 # Compute the Z statistic
#                 # print('Correct answer: ' + str(x[j]))
#                 Z = simple_statistic(0,j,n,p,Y)
#                 # print('Z statistic is: ' + str(Z))
#                 if Z <= 2*lam**2:
#                     xhat_one[j] = xhat_one[0]
#                 else:
#                     xhat_one[j] = (-1)*xhat_one[0]
#                 # print('Final answer: ' + str(xhat[j]))
#                 # print('new bit#########')
#                 # print(xhat_two[0])
#                 # print('Correct answer: ' + str(x[j]))
#                 Z = two_hop_statistic(0,j,n,p,Y,Omega, x)
#                 # print('Z statistic is: ' + str(Z))
#                 if Z >= 0:
#                     xhat_two[j] = xhat_two[0]
#                 else:
#                     xhat_two[j] = (-1)*xhat_two[0]
#                 # print('Final answer: ' + str(xhat_two[j]))
#                 # if xhat_two[j] != x[j]:
#                 #     print('WRONG DECISION######################################')
#                 # print('################')
#         loss_one = np.abs(np.dot(xhat_one, x)) / n
#         loss_two = np.abs(np.dot(xhat_two, x)) / n
#         # print(loss_two)
#         error_50_one[index] += loss_one / num_iters
#         error_50_two[index] += loss_two / num_iters
#     index += 1
# #
#
# n = 100
# error_100_one = np.zeros(len(lam_grid))
# error_100_two = np.zeros(len(lam_grid))
# size_list = []
# for _ in range(p):
#     size_list.append(n)
# # print(size_list)
#
#
# sign = 0
# Omega = np.zeros(size_list)
# it = np.nditer(Omega, flags=['multi_index'])
# while not it.finished:
#     index = it.multi_index
#     Omega[index] = sign
#     sign = 1 - sign
#     it.iternext()
#
# index = 0
# for lam in lam_grid:
#     for i in range(num_iters):
#         x = 2*(np.random.binomial(1, prob, n)-.5)
#         # print(x)
#         Y = np.tensordot(x,x, axes=0)
#         for k in range(p-2):
#             Y = np.tensordot(Y, x, axes=0)
#         Y = lam*Y + np.random.normal(0,1,size_list)
#         # Y = lam*Y
#
#         # print(Y)
#         xhat_one = np.zeros((n))
#         xhat_two = np.zeros((n))
#         # Just set the sign of the first entry to the true one as we only care about signs
#         xhat_one[0] = x[0]
#         xhat_two[0] = x[0]
#
#
#         for j in range(n):
#             if j != 0:
#                 # Compute the Z statistic
#                 # print('Correct answer: ' + str(x[j]))
#                 Z = simple_statistic(0,j,n,p,Y)
#                 # print('Z statistic is: ' + str(Z))
#                 if Z <= 2*lam**2:
#                     xhat_one[j] = xhat_one[0]
#                 else:
#                     xhat_one[j] = (-1)*xhat_one[0]
#                 # print('Final answer: ' + str(xhat[j]))
#                 # print('new bit#########')
#                 # print(xhat_two[0])
#                 # print('Correct answer: ' + str(x[j]))
#                 Z = two_hop_statistic(0,j,n,p,Y,Omega, x)
#                 # print('Z statistic is: ' + str(Z))
#                 if Z >= 0:
#                     xhat_two[j] = xhat_two[0]
#                 else:
#                     xhat_two[j] = (-1)*xhat_two[0]
#                 # print('Final answer: ' + str(xhat_two[j]))
#                 # if xhat_two[j] != x[j]:
#                 #     print('WRONG DECISION######################################')
#                 # print('################')
#         loss_one = np.abs(np.dot(xhat_one, x)) / n
#         loss_two = np.abs(np.dot(xhat_two, x)) / n
#         # print(loss_two)
#         error_100_one[index] += loss_one / num_iters
#         error_100_two[index] += loss_two / num_iters
#     index += 1
#

print(lam_grid)
print(error_25_one)
print(error_25_two)
plt.title('Loss in Estimate vs Signal for p = 3')
plt.xlabel('Signal')
plt.ylabel('Average Accuracy')
plt.plot(lam_grid, error_25_one, label='n = 25, full data')
plt.plot(lam_grid, error_25_two, label='n = 25, two hop')
# plt.plot(lam_grid, error_50_one, label='n = 50, full data')
# plt.plot(lam_grid, error_50_two, label='n = 50, two hop')
# plt.plot(lam_grid, error_100_one, label='n = 100, full data')
# plt.plot(lam_grid, error_100_two, label='n = 100, two hop')
plt.legend()
plt.show()
