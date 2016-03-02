#coding=utf-8
import os
import random

"""
本部分用于测试用梯度法实现多元线性回归算法，而生成的数据集。
这里取4元，也就是 :
Y = B0 + B1 * X1 + B2 * X2 + B3 * X3 + B4 * X4
这里用于生成数据的 [B0, B1, B2, B3, B4] 对应如下：
[10, 1, 2, 3, 4] 即原公式为：
Y = 6 + X1 + 2 * X2 + 3 * X3 + 4 * X4
"""

class Generator(object):

    def generator( self,  num ):

        """
        随机生成num个二元组( X, Y )，对这些数据集进行线性分
        """
        with open( 'multi_grad_data.txt', 'w+' ) as hand:

            i = 1;
            while i <= num :
                """
                我取得随机数始终徘徊在上述公式求出值的附近，这样线性回归分析之后，
                就知道自己的结果是不是正确了
                """
                X1 = random.randint(0, 200)
                X2 = random.randint(0, 200)
                X3 = random.randint(0, 200)
                X4 = random.randint(0, 200)

                B0 = 6 + random.randint(-2,2)
                Y = B0
                Y += X1 + random.randint(-5,5)      # 取这个范围的附加值，使其上下抖动
                Y += 2 * X2 + random.randint(-5,5)
                Y += 3 * X3 + random.randint(-5,5)
                Y += 4 * X4 + random.randint(-5,5)

                hand.write( str(X1) + ' ' + str(X2) + ' ' )
                hand.write( str(X3) + ' ' + str(X4) + ' ' )
                hand.write( str(Y) + '\n' )

                i += 1

        hand.close()

grad_data = Generator()
grad_data.generator(200)   # 生成200个测试数据集
