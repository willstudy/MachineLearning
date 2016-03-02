#coding=utf-8
import os
import random

"""
本部分用于测试用梯度法实现一元线性回归算法，而生成的数据集。
"""

class Generator(object):

    def generator( self,  num ):

        """
        随机生成num个二元组( X, Y )，对这些数据集进行线性分
        """
        with open( 'uni_grad_data.txt', 'w+' ) as hand:

            i = 1;
            while i <= num :
                """
                我取得随机数Y始终徘徊在X值的2倍附近，这样就可以确定线性回归分析之后，
                自己的结果是不是正确了
                """
                X = random.randint(0, 10000)
                Y = 2 * X + random.randint(0, 10)

                hand.write( str(X) + ' ' + str(Y) )
                hand.write( '\n' )

                i += 1

        hand.close()

grad_data = Generator()
grad_data.generator(1000)   # 生成200个测试数据集
