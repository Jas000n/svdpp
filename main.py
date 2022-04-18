import numpy as np
import mv100
list = mv100.mv1002list("/Users/jas0n/PycharmProjects/svd++/ml-100k/u.data")
rm = mv100.creat_matrix(list)
print(rm.shape)
def R(u):  # return the index of rated item for user U
    list = []
    for i in range(0, n):
        if (rm[u][i] != -1):
            list.append(i)
    return list


def sigmaYj(U):
    sum = np.zeros((k, 1))
    for index in R(U):
        sum += y[:, index].reshape(k,1)
    return sum


def getU():  # get average rating
    sum = 0
    total = 0
    for u in range(0, m):
        for i in range(0, n):
            if (rm[u][i] != -1):
                total += 1
                sum += rm[u][i]
    return sum / total
def RMSE():#get rmse loss
    sum =0
    total = 0
    for u in range(0, m):
        for i in range(0, n):
            if (rm[u][i] != -1):
                total +=1
                sum += (rm[u][i] - erm[u][i])**2
    rmse = (sum/total)**0.5
    return rmse


m = rm.shape[0]  # numbers of users
n = rm.shape[1]  # numbers of items
print(m,n)
k = 20  # the length of p & q
p = np.zeros((k, m))  # matrix of user preference
q = np.zeros((n, k))  # matrix of item quality
b_i = np.zeros((1, n))  # item bias
b_u = np.zeros((1, m))  # user bias
y = np.zeros((k, n))  # implicit
epochs = 30  # total epochs
lr = 0.007  # learning rate
decay = 0.9  # decay
l1 = 0.005  # regularization parameter1
l2 = 0.015  # regularization parameter2


erm = np.zeros((m, n))  # estimated rating matrix
average = getU()
print("start calculate erm")
for u in range(0, m):
    # the u-th user
    for i in range(0, n):
        erm[u][i] = average + b_u[0][u] + b_i[0][i] + q[i].reshape(1,k).dot(p[:,u].reshape(k,1) + (1 / len(R(u)) ** 0.5) * sigmaYj(u))
print("end")
for epoch in range(0, epochs):
    for u in range(0, m):
        # the u-th user
        for i in range(0, n):
            # the i-th item
            if (rm[u][i] != -1):
                # update parameter
                print("start")
                eui = rm[u][i] - erm[u][i]
                b_u[0][u] = b_u[0][u] + lr * (eui - l1 * b_u[0][u])
                b_i[0][i] = b_i[0][i] + lr * (eui - l1 * b_i[0][i])
                q[i] = q[i] + lr * (eui * (p[:, u] + 1 / (len(R(u)) ** 0.5) * sigmaYj(u)) - l2 * q[i])
                p[u] = p[u] + lr * (eui * q[i] - l2 * p[u])
                for j in R(u):
                    y[:, j] = y[:, j] + lr * (eui * 1 / (len(R(u)) ** 0.5) * q[i] - l2 * y[:, j])
                print("end")
    lr = lr * decay
    # update estimate rating matrix
    for u in range(0, m):
        # the u-th user
        for i in range(0, n):
            erm[u][i] = average + b_u[0][u] + b_i[0][i] + q[i].reshape(1,k).dot(p[:,u].reshape(k,1) + (1 / len(R(u)) ** 0.5) * sigmaYj(u))
    rmse = RMSE()
    print("this is the {} epoch, the rmse loss is {}".format(epoch+1,rmse))
