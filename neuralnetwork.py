#coding=utf8
from neuralnetworktool import *
import random
import traceback
class Neuron():
	def __init__(self,nInput = 0,studyspeed = 1,thresholdFunc = sigmoidX):
		self.nInput = nInput
		#init the weight in range -1 to 1
		randomweights = [random.uniform(-1, 1) for i in range(nInput)]
		self.weights = randomweights
		self.thresholdFunc = thresholdFunc
		self.studyspeed = studyspeed
		self.inputs = []

	def setInputs(self,inputs):
		self.inputs = inputs

	def getInputs(self):
		return self.inputs

	def getOutput(self,inputs):
		self.inputs = inputs
		outputs = np.squeeze(np.asarray(self.weights * inputs))
		#for i in range(len(self.weights)):
		#	outputs += self.weights[i] * self.inputs[i]
		return round(self.thresholdFunc(outputs),3)


	def adjustWeights(self,inputs,exceptOutput,realOutput = None):
		if realOutput == None:
			f = self.getOutput(inputs)
		else:
			f = realOutput
		d = exceptOutput
		r = (d - f) * f * (1 - f)
		self.adjustWeightsWithR(r)
		return r

	#R反映出网络输出的平方差对相应的sigmoid函数的输入中的变化的敏感度
	def adjustWeightsWithR(self,R):
		self.weights += self.studyspeed * R *  np.transpose(self.inputs)

	def getWeights(self):
		return np.squeeze(np.asarray(self.weights))

	def getWeightsMatrix(self):
		return self.weights

	def setWeights(self,weights):
		self.weights = np.matrix(weights)

	def setStudySpeed(self,studyspeed):
		self.studyspeed = studyspeed

	def printInfo(self):
		print self.weights


class NeuralNetWork():
	def __init__(self,nInput=0,nHide=0,nOutput=0,studyspeed=0):
		self.nInput = nInput
		self.nHide = nHide
		self.nOutput = nOutput
		self.studyspeed = studyspeed
		self.inputNeurons = [0 for i in range(nInput)]
		self.hideNeurons = [
		Neuron(nInput = nInput,studyspeed = studyspeed) for i in range(nHide)]
		self.outputNeurons = [
		Neuron(nInput = nHide,studyspeed = studyspeed) for i in range(nOutput)]


	def backpropagation(self,trainings=[],examples=[]):
		for i in range(len(trainings)):
			self.backpropagationSingle(
				training = trainings[i],
				example = examples[i],
				)

	def backpropagationSingle(self,training,example):
		#d为训练集目标输出
		d = example
		#输入转化为单列矩阵
		self.inputNeurons = np.transpose(np.matrix(training))
		#用于保存隐藏层输出
		hideoutputs = []
		#对输入层，得到隐藏层输出结果
		for hideNeuron in self.hideNeurons:
			hideoutputs.append(hideNeuron.getOutput(self.inputNeurons))
		

		#将隐藏层结果作为输出层的输入，得到输出层的结果
		outputLayeroutputs = []
		hideOutPutMatrixs =  np.transpose(np.matrix(hideoutputs))
		for outputNeuron in self.outputNeurons:
			outputLayeroutputs.append(outputNeuron.getOutput(hideOutPutMatrixs))
		#输出层结果与期望结果做比较，并且反向传播调节
		#保存调节前输出层权值
		beforeweights = []

		#对输出层进行调节
		#保存各个输出层的调节因子
		outputRs = []
		for i in range(self.nOutput):
			#f为实际输出结果
			f = outputLayeroutputs[i]
			outputR = (d[i] - f) * f * (1 - f)
			outputRs.append(outputR)
			#保存第i个输出神经元的调节前权值
			beforeweights.append(self.outputNeurons[i].getWeights())
			#对每个输出层神经元进行调节
			self.outputNeurons[i].adjustWeightsWithR(outputR)
		#对隐藏层进行调节
		for i in range(self.nHide):
			rw = 0
			for j in range(self.nOutput):
				rw += outputRs[j] * beforeweights[j][i]
			#计算隐藏层该神经元的调节因子
			hideR = hideoutputs[i] * (1 - hideoutputs[i]) * rw
			#对第j个隐藏层神经元调节
			self.hideNeurons[i].adjustWeightsWithR(hideR)

	def getOutput(self,inputs):
		hideoutputs = []
		inputMatrix = np.transpose(np.matrix(inputs))
		#一输入层为输入计算每个隐藏层神经元输出，得到隐藏层输出结果
		for hideNeuron in self.hideNeurons:
			hideoutputs.append(hideNeuron.getOutput(inputMatrix))

		outputLayeroutputs = []
		#将隐藏层结果作为输出层的输入，得到输出层的结果
		hideoutpuMatrix = np.transpose(np.matrix(hideoutputs))
		for outputNeuron in self.outputNeurons:
			outputLayeroutputs.append(outputNeuron.getOutput(hideoutpuMatrix))
		return outputLayeroutputs

	def getHideNeurons(self):
		return self.hideNeurons

	def setHideNeurons(self,arg):
		self.hideNeurons = arg

	def getOutputNeurons(self):
		return self.outputNeurons

	def setOutputNeurons(self,arg):
		self.outputNeurons = arg

	def getSaveHideNeurons(self):
		res = []
		for hideNeuron in self.getHideNeurons():
			res.append( ndarray.tolist(hideNeuron.getWeights()) )
			return res

	def getSaveOutPutNeurons(self):
		res = []
		for outputNeuron in self.getOutputNeurons():
			res.append( ndarray.tolist(outputNeuron.getWeights()))
			return res

	def getSaveData(self):
		saveData = {}
		saveData['nInput'] = self.nInput
		saveData['nHide'] = self.nHide
		saveData['nOutput'] = self.nOutput
		saveData['studyspeed'] = self.studyspeed
		saveData['hideNeurons'] = self.getSaveHideNeurons()
		saveData['ouputNeurons'] = self.getSaveOutPutNeurons()
		return saveData

	def setBySaveData(self,savedata):
		self.nInput = savedata['nInput']
		self.nHide = savedata['nHide']
		self.nOutput = savedata['nOutput']
		self.studyspeed = savedata['studyspeed']
		self.inputNeurons = [0 for i in range(nInput)]
		self.hideNeurons = [
		Neuron(nInput = nInput,studyspeed = studyspeed) for i in range(nHide)]
		self.outputNeurons = [
		Neuron(nInput = nHide,studyspeed = studyspeed) for i in range(nOutput)]
		#设定隐藏层权重
		hideDatas = savedata['hideNeurons']
		for i in range(nHide):
			self.hideNeurons[i].setWeights(np.matrix(hideDatas[i]))
		outputDatas = savedata['ouputNeurons']
		for i in range(nOutput):
			self.outputNeurons[i].setWeights(np.matrix(outputDatas[i]))

	def printInfo(self):
		print 'hide'
		for hideNeuron in self.hideNeurons:
			hideNeuron.printInfo()
		print 'output'
		for outputNeuron in self.outputNeurons:
			outputNeuron.printInfo()


def Debug():
	# neuralnetworkInstance = NeuralNetWork(3,2,1,1)
	# trainings = [
	# [1,0,1],
	# [0,0,1],
	# [0,1,1],
	# [1,1,1]
	# ]
	# examples = [
	# 0,1,0,1
	# ]
	# for i in range(500):
	# 	neuralnetworkInstance.backpropagation(trainings,examples)
	# 	print neuralnetworkInstance.getOutput([1,0,1])
	N1 = Neuron(3,1)
	N1.setWeights([2,-2,0])
	N2 = Neuron(3,1)
	N2.setWeights([1,3,-1])
	# print N1.getOutput([1,0,1])
	# print N2.getOutput([1,0,1])
	N4 = Neuron(3,1)
	N4.setWeights([1,1,1])

	N3 = Neuron(3,1)
	N3.setWeights([3,-2,-1])
	# print N3.getOutput([N1.getOutput([1,0,1]),N2.getOutput([1,0,1]),1])
	# print N3.getWeights()
	# print N3.adjustWeights([N1.getOutput([1,0,1]),N2.getOutput([1,0,1]),1],0)
	# print N3.getInputs()
	# print N3.getWeights()


	d = 0
	hideoutputs = []
	#对输入层，得到隐藏层输出结果
	hideoutputs.append(N1.getOutput(np.transpose(np.matrix([1,0,1]))))
	hideoutputs.append(N2.getOutput(np.transpose(np.matrix([1,0,1]))))
	hideoutputs.append(N4.getOutput(np.transpose(np.matrix([1,0,1]))))
	print 'hideoutputs',hideoutputs

	outputLayeroutputs = []
	#将隐藏层结果作为输出层的是输入，得到输出层的结果
	outputLayeroutputs.append(N3.getOutput(np.transpose(np.matrix((hideoutputs)))))
	#输出层结果与期望结果做比较，并且反向传播调节
	#f为实际输出结果
	f = outputLayeroutputs[0]
	print 'output',f
	outputR = (d - f) * f * (1 - f)
	#对每个输出层神经元进行调节
	print 'outputR',outputR
	beforeweights = N3.getWeights()
	print 'beforeweights',beforeweights
	N3.adjustWeightsWithR(outputR)
	print 'afterweights',N3.getWeights()
	afterweights = N3.getWeights()
	rw = afterweights[0] * outputR
	print 'rw',rw
	#对隐藏层进行调节
	hideR = hideoutputs[0] * (1 - hideoutputs[0]) * rw
	#对第j个隐藏层神经元调节
	print "N1before",N1.getWeights()
	N1.adjustWeightsWithR(hideR)
	print "N1before",N1.getWeights()


	rw = afterweights[1] * outputR
	print 'rw',rw
	hideR = hideoutputs[1] * (1 - hideoutputs[1]) * rw
	#对第j个隐藏层神经元调节
	print "N2before",N2.getWeights()
	N2.adjustWeightsWithR(hideR)
	print "N2before",N2.getWeights()

	rw = afterweights[2] * outputR
	print 'rw',rw
	hideR = hideoutputs[2] * (1 - hideoutputs[2]) * rw
	print "Nbefore",N4.getWeights()
	N4.adjustWeightsWithR(hideR)
	print "N4before",N4.getWeights()

	print '\n\n\n\n\n\n'
	net = NeuralNetWork(nInput=3,nHide=3,nOutput=1,studyspeed=1)

	d = 0
	examples = [[0]]
	N11 = Neuron(3,1)
	N11.setWeights([2,-2,0])
	N21 = Neuron(3,1)
	N21.setWeights([1,3,-1])
	N31 = Neuron(3,1)
	N31.setWeights([1,1,1])
	N41 = Neuron(3,1)
	N41.setWeights([3,-2,-1])
	hides =  [N11,N21,N31]
	outputs = [N41]
	net.setHideNeurons(hides)
	net.setOutputNeurons(outputs)



	net.backpropagation([[1,0,1]],[[0]])
	net.printInfo()

def xulian(times,nInput,nHide,nOutput):
	try:
		data_file = open('img.json','r')
		res_file = open('res_%d_%d_%d_%d'%(times,nInput,nHide,nOutput),'w+')
		json_data = eval(data_file.read())
		#获得训练集
		imgs = json_data['imgs']
		trainingimgs = imgs[:300]
		testimgs = imgs[301:]
		print len(imgs)
		#定义神经网络
		network = NeuralNetWork(nInput = nInput , nHide = nHide,nOutput = nOutput,studyspeed = 0.05)

		#训练过程
		for time in range(times):
			count = 0
			for img in trainingimgs:
				training = img['data']
				if img['glass'] == 'open':
					example = [1]
				else:
					example = [0]
				network.backpropagationSingle(training = training,example = example)
				print count
				count += 1
				print 'success training time:%d item:%d'%(time,count)
			#测试
			count = 0
			success = 0
			for img in testimgs:
				testItem = img['data']
				if img['glass'] == 'open':
					example = [1]
				else:
					example = [0]
				print 'testid',count,':',example
				res =  network.getOutput(testItem)
				print res
				if round(res[0],0) == example[0]:
					success += 1
				count += 1
			print 'Trainings times %d : %f'%(time ,float(success)/len(testimgs))
			res_file.write('[%d,%f]\n'%(time ,float(success)/len(testimgs)))
		#保存训练结果
		##save_file = open('net%s.json'%times,'w')
		##save_file.write(str(network.getSaveData()))
		#print network.getSaveData()['hideNeurons']

		#测试
		
		
	except Exception as e:
		print traceback.format_exc()


def test(times):
	data_file = open('img.json','r')
	json_data = eval(data_file.read())
	#根据配置文件设置神经网络
	network = NeuralNetWork()
	#获取配置文件
	confdig_file = open('net%s.json'%times,'r')
	json_config = eval(confdig_file.read())
	network.setBySaveData(json_config)
	#获得训练集
	imgs = json_data['imgs']
	for img in imgs:
		testItem = img['data']
		if img['glass'] == 'open':
			example = [1]
		else:
			example = [0]
		print 'testid',count,':',example
		res =  network.getOutput(testItem)
		print res
		if round(res[0],0) == example[0]:
			success += 1
		count += 1
	print 'success for times %d :'%times ,success/624

def main():
	#Debug()
	xulian(150,3840,15,1)
	
if __name__ == '__main__':
	main()
	

