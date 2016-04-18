import os
import sys, getopt

way = ""
model = ""
max_iter = 100

opts, args = getopt.getopt(sys.argv[1:],"ecim::s::",["e", "ec", "if", "model=", "step="] )

for name, value in opts:
	if name in ("-e", "--e"):
		way = 0
		print "ESTIMATE"
	if name in ("-c", "--ec"):
		way = 1
		print "ESTIMATEC"
	if name in ("-i", "--if"):
		way = 2
		print "INFERENCE"
	if name in ("-m", "--model"):
		model = value
		print "model : %s" % model 
	if name in ("-s", "--step"):
		max_iter = value
		print "max_iter : %d " % int(max_iter)

