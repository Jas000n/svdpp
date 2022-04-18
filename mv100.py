
import random

import  numpy as np
import torch
from torch import nn



def get_U_from_list(list):
    '''
    :param list:  一个放有数据集的列表: "userid merchandise score timestomp"
    :return:
    '''
    U=0.0
    for i in list:
        U+=i[2]
    return U/float(len(list))



def str2float(x):
    return float(x)

def turncomment2data(str):
    '''
    :param str: 一个放有数据集的列表，格式如下: "userid merchandise score timestomp"
    :return:4 float data
    '''
    # print(str.split('\t'))
    return (list(map(str2float,str.split('\t'))))



def mv1002list(filename):
    '''
    :param filename:数据的文件位置
    :return:一个存放mv100的列表
    '''
    mvlist=[]

    f=open(filename)
    while True:

        text=f.readline()

        if text=='':
            break

        mvlist.append(turncomment2data(text[:-2]))
        # print(text)
        # counter+=1
    # print(mvlist)
    ma=np.array(mvlist)
    # print(ma.shape)
    f.close()
    return mvlist

def creat_matrix(list):
    '''
    :param list: 一个放有数据集的列表，自动寻找矩阵的长宽,所以需要寻找最大的x和y
    :return: 一个完全的矩阵
    '''
    max_width=0
    max_hight=0
    for i in list:
        if i[0]>max_hight:
            max_hight=i[0]
        if i[1]>max_width:
            max_width=i[1]
    '''
    这个地方初始化只要不在评分范围内就行，没有评分的不计算梯度就可以啦
    '''
    array=np.full((int(max_hight),int(max_width)),-1)

    for i in list:
        # print(i)
        # to ensure the consistence of data I add 1 to array size
        array[int(i[0]-1),int(i[1])-1]=i[2]
        # print(array[int(i[0])-1,int(i[1])-1])
    return array



