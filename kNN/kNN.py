#coding=utf-8
from numpy import *
import operator

"""
这是kNN K-邻近算法的实现，这个算法思想比较简单，准确度很高，但计算量很大。
数据集是我从《机器学习实战》上复制的，现在自己动手实现以下这个代码

数组的shape()会返回一个包含行列数的列表，shape[0]表示行数，shape[1]表示列数

tile([1,2,3],(2,1)) 会返回
[[1,2,3],
 [1,2,3]]

数组的min和max方法会返回每一列中最小和最大的元素
数组的sum(axis=1)会是数组的每一行的元素进行相加，最后形成n行1列的矩阵
数组的argsort方法会返回数组值从小到大的索引
>>> x = np.array([3, 1, 2])
>>> np.argsort(x)
array([1, 2, 0])
"""

class k_NN(object):

    def load_set( self, filename ):

        input_materix = []
        Y_set = []

        with open( filename, 'r' ) as file_hand:

            for line in file_hand.readlines():
                X_set = []
                line = line.strip().split('\t')
                for item in line:
                    X_set.append( float(item) )

                Y_set.append( X_set.pop() )
                input_materix.append( X_set )

        return input_materix, Y_set

    def auto_norm( self, input_materix ):           # 对输入矩阵进行归一化处理

        min_val = input_materix.min(0)              # 取每列的最小一个数据，组成的一行
        max_val = input_materix.max(0)              # 取每列的最大一个数据，组成的一行

        ranges = max_val - min_val
        num_line = input_materix.shape[0]

        norm_set = input_materix - tile(min_val,(num_line,1))
        norm_set = norm_set / tile(max_val,(num_line,1))

        return norm_set, ranges, min_val

    """
    user_input: 用户输入关于各个特征量的值，来判断它属于哪个标签
    """
    def knn( self, filename, user_input, k ):

        input_materix, Y_set = self.load_set( filename )
        input_materix = array(input_materix)

        norm_set, ranges, min_val = self.auto_norm( input_materix )
        user_input = (user_input - min_val) / ranges      # 对输入的数据进行归一化

        num_line = input_materix.shape[0]          # 样本数据共多少个，多少行
        user_input = tile( user_input, (num_line, 1) )

        offset = user_input - input_materix
        distance = (offset ** 2).sum(axis=1)            # 矩阵的每一行进行相加
        distance = distance ** 0.5
        sort_distance = distance.argsort()              # 排序后的下标

        lable_count = {}
        for index in range(k):
            lable = Y_set[sort_distance[index]]           # 得到它的标签
            lable_count[lable] = lable_count.get(lable,0) + 1   # 得到这个标签的值，若之前没有这个标签则设为0

        sort_lable = sorted( lable_count.iteritems(), lambda d:d[1], reverse = True )
        return sort_lable[0][0]

"""
我从data.txt中随便去一个数据进行测试的。。
14488	7.153469	1.673904	2
"""
user_input = [14488,7.153469,1.673904]
user_input = array(user_input)

knn = k_NN()
lable = knn.knn( 'data.txt', user_input, 3 )

print lable

"""
输出为 2.0
"""
