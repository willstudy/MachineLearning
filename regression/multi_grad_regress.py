#coding=utf-8
import os
import math
import array
from numpy import *
from numpy.linalg import inv

"""
用普通最小二乘法简单的实现多元线性回归思想，主要通过
( X1, X2, X3, X4, Y )数据集合，求出Y与(X1, X2, X3, X4)的线性关系，即

Y = B0 + B1 * X1 + B2 * X2 + B3 * X3 + B4 * X4

求出 [B0, B1, B2, B3, B4]

主要根据线性代数的公式：

transpo([B0, B1, B2, B3, B4]) = reverse((transpo(X) * X )) * transpo(X) * Y

transpo代表矩阵转置，reverse代表矩阵求逆

X 代表 (X1, X2, X3, X4)的输入矩阵
Y 代表 输出矩阵

"""

class LinearRegression( object ):

    def __init__( self ):

        self.BO = 0;
        self.B1 = 0;
        self.B2 = 0;
        self.B3 = 0;
        self.B4 = 0;

    def load_data( self, filename ):

        input_set = []     # X 的输入集合
        Y_set = []         # Y 的输出集合

        with open( filename, 'r' ) as hand :
            for line in hand.readlines():

                line_data = line.strip('\n').split(' ')
                X_set = []      # ( X1, X2, X3, X4 )

                X_set.append(1.0)      # 增设变量 BO，对应常量项
                for item in line_data :
                    X_set.append(float(item))

                Y_set.append( X_set.pop() )
                input_set.append( X_set )

        hand.close()
        return input_set, Y_set

    def regress( self, filename ):
        """
        根据上面写的公式，求出B的矩阵结果
        """
        input_set, Y_set = self.load_data(filename)
        input_matrix = array( input_set )     # 转换成输入矩阵
        # 获得输入矩阵的转置
        input_matrix_transpose = input_matrix.transpose()
        # 将Y输入矩阵转成 len(Y_set)行 1列的矩阵
        Y_matrix = array( Y_set ).reshape( len(Y_set), 1 )
        reverse_result = inv( dot(input_matrix_transpose, input_matrix) )  # 与转置矩阵相乘的乘积

        B_matrix = dot( dot( reverse_result , input_matrix_transpose ), Y_matrix )

        print B_matrix

        return B_matrix

regress = LinearRegression()
regress.regress('multi_grad_data.txt')

"""
我机器上输出结果为：
[[ 4.10575543]
 [ 1.00000618]
 [ 2.00396198]
 [ 3.00571856]
 [ 4.0019599 ]]
这样的结果符合预期，最小二乘法求多元线性回归太给力了！！！！哈哈哈~~
"""
