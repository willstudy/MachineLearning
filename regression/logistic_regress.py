#coding=utf-8
import os
import math
from numpy import *
import matplotlib.pyplot as plt

"""
这些代码是依据《机器学习实战》写出来的，我现在还不是很理解
回归系数（最大似然）怎么求得？？
回归系数相当于每个特征向量的比重，我是这么理解的。。。。

看了很多博主的文章，都只推导了回归系数的公式，但为什么代码使用
输入矩阵的转置 * 输出矩阵的偏差 来 代表对回归系数的偏导呢？？

这次没有自己动手生成数据集，因为我还没弄明白这个过程，
而是使用以为博主的数据集，放在了logistic_sample.txt中
输入：( X1, X2 )  输出：( Y ) Y非0即1

代码中使用input_matrix表示特征向量矩阵，Y_set代表输出矩阵

下面的代码是我借鉴了一些博主，然后加了一点自己的理解。
"""

class Logistic(object):

    def load_set( self, filename ):

        input_matrix = []               # 特征向量的输入矩阵
        X_set = []                      # 每一行的特征向量
        Y_set = []                      # 类别（0，1）的输出矩阵

        with open(filename, 'r') as hand:

            for line in hand.readlines():
                line_data = line.strip('\n').split(' ')
                X_set = []              # 每一行的特征向量
                X_set.append(1.0)
                for item in line_data:
                    X_set.append( float(item) )

                Y_set.append( X_set.pop() )
                input_matrix.append( X_set )

        hand.close()
        return input_matrix, Y_set

    """
    这是一位博主写的图形化显示程序
    """
    def showLogRegres(self, weights, train_x, train_y):
        # notice: train_x and train_y is mat datatype
        numSamples, numFeatures = shape(train_x)
        if numFeatures != 3:
            print "Sorry! I can not draw because the dimension of your data is not 2!"
            return 1

        # draw all samples
        for i in xrange(numSamples):
            if int(train_y[i, 0]) == 0:
                plt.plot(train_x[i, 1], train_x[i, 2], 'or')
            elif int(train_y[i, 0]) == 1:
                plt.plot(train_x[i, 1], train_x[i, 2], 'ob')

        # draw the classify line
        min_x = min(train_x[:, 1])[0, 0]
        max_x = max(train_x[:, 1])[0, 0]
        weights = weights.getA()  # convert mat to array
        y_min_x = float(-weights[0] - weights[1] * min_x) / weights[2]
        y_max_x = float(-weights[0] - weights[1] * max_x) / weights[2]
        plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')
        plt.xlabel('X1'); plt.ylabel('X2')
        plt.show()

    def sigmoid( self, input_X ):

        return  1.0 / ( 1 + exp(-input_X) )

    """
    filename : 数据集的文件名
    method   : 求回归系数的方法
    """
    def logistic_regress( self, filename, method ):

        input_matrix, Y_set = self.load_set( filename )
        input_matrix = mat(input_matrix)      # 转成矩阵的形式
        Y_set = mat(Y_set).transpose()        # 转成矩阵后，变换成它的转置矩阵

        num_sample, num_feature = shape( input_matrix )         # 获得样本数据的个数和特征值的个数

        alpha = 0.01            # 初始化步长
        max_inter = 200         # 最大迭代次数
        weight = ones( (num_feature, 1) )  #初始化回归系数矩阵，即生成一个num_feature行 1列的矩阵，其值为1

        if method == "grad_descent":        # 普通的梯度下降法就回归系数
            for it in range( max_inter ):
                result = self.sigmoid( input_matrix * weight )  # 使用此回归系数，求出的特征向量的输出值
                offset = Y_set - result         # 与实际结果的偏移
                weight = weight + alpha * input_matrix.transpose() * offset   # 更新回归系数

        if method == "random_grad_descent":      # 随机梯度下降方法
            for index in range(num_sample):
                result = self.sigmoid( input_matrix[index] * weight )
                offset = Y_set[index] - result
                weight = weight + alpha * input_matrix[index].transpose() * offset

        if method == "opt_random_descent":      # 改进后的随机梯度下降法
            for it in range( max_inter ):       # 迭代没有收敛？？每次效果都不一样。。。
                data_index = range(num_sample)
                for index in range( num_sample ):
                    alpha = 4.0 / ( 1.0 + index + it ) + 0.01
                    rand_index = int(random.uniform(0, len(data_index)))    # 生成一个随机的下标
                    result = self.sigmoid( input_matrix[rand_index] * weight )
                    offset = Y_set[rand_index] - result
                    weight = weight + alpha * input_matrix[index].transpose() * offset
                    del(data_index[rand_index])     # 删除这个下标

        self.showLogRegres( weight, input_matrix, Y_set )

        return weight


logistic = Logistic()
weight = logistic.logistic_regress('logistic_sample.txt', "opt_random_descent")
#print weight
"""
grad_descent ：
[[ 11.75683137]
 [  1.30112337]
 [ -1.37079338]]

这个结果下：表达式即为
0 = 11.756.. + 1.3011.. * X1 - 1.370.. * X2

random_grad_descent ：
[[ 1.01702007]
 [ 0.85914348]
 [-0.36579921]]

opt_random_descent :
[[-10.2776162 ]
 [-21.03483329]
 [  0.53641868]]
"""
