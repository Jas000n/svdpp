import time

import numpy as np
import mv100

train_list = mv100.mv1002list("/Users/jas0n/PycharmProjects/svd++/ml-100k/u1.base")
train_length = len(train_list)
rm = mv100.creat_matrix(train_list)
test_list = mv100.mv1002list("/Users/jas0n/PycharmProjects/svd++/ml-100k/u1.test")
print(rm.shape)




def getU():  # get average rating
    sum = 0
    total = 0
    for u in range(0, m):
        for i in range(0, n):
            if (rm[u][i] != -1):
                total += 1
                sum += rm[u][i]
    return sum / total


def RMSE(test):  # get rmse loss
    sum = 0
    total = 0
    for line in test:
        u_id = int(line[0]) - 1
        i_id = int(line[1]) - 1
        score = line[2]
        sum += (score - Eui(u_id, i_id)) ** 2
        total += 1
    rmse = (sum / total) ** 0.5
    return rmse


def MAE(test):  # get mae loss
    sum = 0
    total = 0
    for line in test:
        u_id = int(line[0]) - 1
        i_id = int(line[1]) - 1
        score = line[2]
        sum += abs(score - Eui(u_id, i_id))
        total += 1
    mae = sum / total
    return mae


def Eui(u, i):  # return the estimate value of the specific user u on item i
    eui = average + b_i[i] + b_u[u]

    return eui


m = rm.shape[0]  # numbers of users
n = rm.shape[1]  # numbers of items

k = 50  # the length of p & q
p = np.zeros((k, m))  # matrix of user preference
q = np.zeros((n, k))  # matrix of item quality
b_i = [0] * n  # item bias
b_u = [0] * m  # user bias
y = np.zeros((k, n))  # implicit
epochs = 30  # total epochs
lr = 0.007  # learning rate
decay = 0.9  # decay
l1 = 0.005  # regularization parameter1
l2 = 0.015  # regularization parameter2

rmse_result_list = []
mae_result_list = []

# erm = np.zeros((m, n))  # estimated rating matrix
average = getU()

times = 0
total_trainning_time = 0
total_test_time = 0
rmse = RMSE(test_list)
mae = MAE(test_list)
print("after initializing,\tthe rmse loss is {},\tthe mae loss is {}".format( rmse, mae))
for epoch in range(0, epochs):
    current_epoch_times = 0
    for line in train_list:
        time1 = time.time()
        # every single line of the rating data
        # update parameter
        u_id = int(line[0]) - 1
        i_id = int(line[1]) - 1
        score = line[2]
        eui = score - Eui(u_id, i_id)
        b_u[u_id] = b_u[u_id] + lr * (eui - l1 * b_u[u_id])
        b_i[i_id] = b_i[i_id] + lr * (eui - l1 * b_i[i_id])

        time2 = time.time()
        times += 1
        current_epoch_times+=1
        total_trainning_time += (time2 - time1)
        if (current_epoch_times % 1000 == 0):
            print(
                "has step {} times,\t{}%,\testimated epoch time is {} s".format(current_epoch_times, round(current_epoch_times / train_length * 100,2),
                                                                        total_trainning_time / times * train_length))
    lr = lr * decay

    rmse = RMSE(test_list)
    mae = MAE(test_list)
    print("this is the {} epoch,\tthe rmse loss is {},\tthe mae loss is {}".format(epoch + 1, rmse,mae))
    time3 = time.time()
    rmse_result_list.append(rmse)
    mae_result_list.append(mae)
    time4 = time.time()
    total_test_time += (time4-time3)
print("rmse",rmse_result_list)
print("=======================================")
print("mae",mae_result_list)
print("the total_time used in trainning is {}, the total test time is {}".format(total_trainning_time,total_test_time))
