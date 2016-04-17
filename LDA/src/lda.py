#coding=utf-8

from __future__ import division

import os
import random

class LDA:

	def __init__( self ):
		# 单词到ID的映射表	
		self.word_map = {}
		# ID到单词的映射	
		self.id2word = []
		# 保存每篇文章的词数
		self.doc_word_num = []
		# 保存每篇文章
		self.doc = []
		# 保存每篇文章的每个单词对应的主题
		self.doc_word_topic = []
		# 保存每个主题的词数
		self.topic = []
		# 保存每篇文章的主题单词个数
		self.doc_topic = []
		# 这个词属于哪个主题
		self.word_id_topic = []
		# 文档--主题的概率分布
		self.theta = []
		# 主题--单词的概率分布
		self.phi = []
		# 样本中文档的个数
		self.M = 0
		# 样本中单词的个数
		self.V = 0
		# 样本中主题的个数
		self.K = 0
		# 文档——主题的Dirichlet分布参数
		self.alpha = 0
		# 主题——单词的Dirichlet分布参数
		self.beta = 0
		# 最大迭代次数
		self.max_iter = 0

	def initial( self, sample_file, max_iter = 100, alpha = 0.5, beta = 0.1, topic_num = 20 ):

		self.max_iter = max_iter
		self.alpha = alpha
		self.beta = beta
		self.K = topic_num

		self.M = int(sample_file.readline().strip())
		
		for line in sample_file.readlines():

			line = line.strip().split(' ')
			for word in line:
				if self.word_map.has_key(word):
					pass
				else :
				 	self.word_map[word] = len(self.word_map) 
				 	self.id2word.append(word)
			# 保存每篇文章
			self.doc.append(line)
			# 保存每篇文章的单词个数
		  	self.doc_word_num.append(len(line))	  
		
		self.V = len(self.word_map)
		
		"""
		初始化每个主题中含有的单词数
		"""
		k = 0
		while k < self.K :
			self.topic.append(0)
			k += 1
		"""
		初始化每个单词ID属于不同主题的个数
		"""
		word_id = 0
		while word_id < self.V:
			word_topic = []
			topic_index = 0
			while topic_index < self.K:
				word_topic.append(0)
				topic_index += 1
			self.word_id_topic.append(word_topic)
			word_id += 1
		"""
		初始化每篇文章，每个主题含有的单词数
		"""
		doc_num = len(self.doc)
		
		doc_id = 0
		while doc_id < doc_num:
			topic_word = []
			topic_index = 0
			while topic_index < self.K:
				topic_word.append(0)
				topic_index += 1
			self.doc_topic.append(topic_word)
			doc_id += 1

		"""
		随机为这些单词，赋值一个主题
		"""
		doc_id = 0
		while doc_id < doc_num :
			document = self.doc[doc_id]
			word_num = self.doc_word_num[doc_id]
			word_index = 0
			word_topic = []
			while word_index < word_num:
				random_topic = random.randint(0, self.K-1)
				word_id = self.word_map[document[word_index]]
				self.word_id_topic[word_id][random_topic] += 1
				self.topic[random_topic] += 1
				self.doc_topic[doc_id][random_topic] += 1
				#  每个单词对应的主题
				word_topic.append(random_topic)
				word_index += 1
			# 每篇文章m的第n个单词对应的主题 doc_word_topic[m][n] = random_topic
			self.doc_word_topic.append(word_topic)
			doc_id += 1

	def sampling( self, doc_id, word_index ):
			word_id = self.word_map[self.doc[doc_id][word_index]]
			topic = self.doc_word_topic[doc_id][word_index]
			self.doc_topic[doc_id][topic] -= 1
			self.topic[topic] -= 1
			self.word_id_topic[word_id][topic] -= 1
			self.doc_word_num[doc_id] -= 1
			
			vbeta = self.V * self.beta
			kalpha = self.K * self.alpha

			k = 0
			pro = []
			while k < self.K:
				"""
				Gibbs Sampling 推导的抽样公式
				"""
				probility = ( self.word_id_topic[word_id][k] + self.beta ) \
						/ ( self.topic[k] + vbeta ) \
						* ( self.doc_topic[doc_id][k] + self.alpha ) \
						/ ( self.doc_word_num[doc_id] + kalpha )

				pro.append(probility)
				k += 1
			"""
			掷筛子算法
			"""
			k = 1
			while k < self.K:
				pro[k] += pro[k-1]
				k += 1
			
			possible = random.random() * pro[self.K - 1]

			topic = 0
			while topic < self.K:
				if pro[topic] > possible :
					break
				topic += 1

			self.word_id_topic[word_id][topic] += 1
			self.doc_topic[doc_id][topic] += 1
			self.topic[topic] += 1
			self.doc_word_num[doc_id] += 1

			return topic

	def compute_theta( self ):
		doc_id = 0
		while doc_id < self.M:
			topic_index = 0
			theta_pro = []
			while topic_index < self.K:
				temp_pro = ( self.doc_topic[doc_id][topic_index] + self.alpha ) / \
					   ( self.doc_word_num[doc_id] + self.K * self.alpha )
				theta_pro.append(temp_pro)
				topic_index += 1
			self.theta.append(theta_pro)
			doc_id += 1
	
	def compute_phi( self ):
		topic_index = 0
		while topic_index < self.K:
			word_id = 0
			phi_pro = []
			while word_id < self.V:
				temp_pro = ( self.word_id_topic[word_id][topic_index] \
						+ self.beta ) / ( self.topic[topic_index] \
						+ self.V * self.beta )	 
				phi_pro.append(temp_pro)
				word_id += 1
			self.phi.append(phi_pro)
			topic_index += 1
	
	def record_model( self ):
		os.chdir('../model')

		theta_file = open('theta.txt', 'w+')
		phi_file = open('phi.txt', 'w+')
		assign_file = open('assign.txt', 'w+')
		topic_word_file = open('topic_word.txt', 'w+')
		
		doc_id = 0 
		while doc_id < self.M:
			for item in self.theta[doc_id] :
				theta_file.write( str(item) + ' ' )	
			doc_id += 1
			theta_file.write('\n')	

		theta_file.close()
		
		topic_index = 0
		while topic_index < self.K:
			for item in self.phi[topic_index]:
				phi_file.write( str(item) + ' ')
			topic_index += 1
			phi_file.write('\n')

		phi_file.close()
		
		doc_id = 0
		while doc_id < self.M:
			word_num = self.doc_word_num[doc_id]
			word_index = 0
			while word_index < word_num:
				topic = self.doc_word_topic[doc_id][word_index]
				assign_file.write(str(self.doc[doc_id][word_index]) + ':' + str(topic) + ' ')
				word_index += 1
			assign_file.write('\n')
			doc_id += 1
		
		assign_file.close()
		
		topic_word = {}
		word_id = 0
		while word_id < self.V:
			topic_index = 0
			while topic_index < self.K:
				num = self.word_id_topic[word_id][topic_index]
				word = self.id2word[word_id]
				if topic_word.has_key(topic_index):
					pair = topic_word[topic_index]
					if pair.has_key(word):
						pair[word] += num
					else:
					 	pair[word] = num
				else:
				  	pair = {}
					pair[word] = num
					topic_word[topic_index] = pair
				topic_index += 1
			word_id += 1

		topic_index = 0
		while topic_index < self.K:
			topic_word_file.write(str(topic_index)+': \n')
			word_list = topic_word[topic_index]
			word_list = sorted( word_list.iteritems(), key=lambda a:a[1], reverse=True )

			i = 0
			for key,value in word_list:
				if i > 10:
					break
				topic_word_file.write(str(key) + ' ')
				i += 1

			topic_word_file.write('\n')
			topic_index += 1

		topic_word_file.close()
			
	def estimate( self ):
		
		init_iter = 0

		while init_iter < self.max_iter :
			print init_iter
			doc_id = 0
			while doc_id < self.M:
				document = self.doc[doc_id]
				word_num = self.doc_word_num[doc_id]
				word_index = 0
				while word_index < word_num:
					topic = self.sampling( doc_id, word_index )
					self.doc_word_topic[doc_id][word_index] = topic
					word_index += 1
				doc_id +=1
			init_iter += 1

		self.compute_theta()
		self.compute_phi()
		self.record_model()	

lda = LDA()

with open('../model/sample.txt', 'r') as file_read:
	
	lda.initial(file_read)
	lda.estimate()
