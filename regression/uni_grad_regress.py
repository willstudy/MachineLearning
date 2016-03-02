#coding=utf-8
import os
import math

"""
用梯度法简单的实现线性回归思想，主要通过
( X, Y )数据集合，求出Y与X的线性关系，即

Y = a * X + c

求出 a, c

先定义 a, c 的取值区间，这里因为练手，我随便取的，分别为
(-10, 10)，(-1000,-1000) 递增的梯度为 0.1，1

"""

class LinearRegression( object ):

    def __init__( self ):

        self.a = 0;
        self.c = 0;

    def load_data( self, filename ):

        data_set = []

        with open( filename, 'r' ) as hand :

            for line in hand.readlines():

                line_data = line.strip('\n').split(' ')
                data_set.append( line_data )

        hand.close()
        return data_set

    def cal_cost( self, data_set, A, C ):

        cost = 0.0
        realY = 0.0
        predicY = 0.0
        offset = 0

        for line_data in data_set :

            X = float(line_data[0])
            realY = float(line_data[1])
            predicY = A * X + C

            offset = realY - predicY
            cost += offset * offset

        return cost

    def regress( self, filename ):

        data_set = self.load_data(filename)

        A = -10
        C = -1000
        min_cost = self.cal_cost( data_set, A, C )

        while A < 10 :

            cost = self.cal_cost( data_set, A, self.c )
            if cost < min_cost :
                min_cost = cost
                self.a = A
            A += 0.1

        while C < 1000 :
            # 利用最优的 a 系数，求 C的值
            cost = self.cal_cost( data_set, self.a, C )
            if cost < min_cost :
                min_cost = cost
                self.c = C
            C += 1

    def get_result( self ):

        result = []
        result.append(self.a)
        result.append(self.c)

        return result

regress = LinearRegression()
regress.regress('grad_data.txt')
result = regress.get_result()
print result

"""
我机器上输出结果为： a = 1.999999999 c = 5  这样的结果符合预期
"""
