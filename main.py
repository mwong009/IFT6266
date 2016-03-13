# Code originally created by Melvin Wong
# Source: https://github.com/mwong009/IFT6266
import six.moves.cPickle as pickle
import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wave
from lstm import LSTM

from fuel.datasets.youtube_audio import YouTubeAudio
from fuel.transformers.sequences import Window

from optparse import OptionParser

# Parse options
parser = OptionParser()
parser.add_option("-n", "--n-hidden", help="Number of hidden units",
								 action="store", type="int", dest="hiddenUnits", default=250)
parser.add_option("-f", "--freq", help="Frequency",
								 action="store", type="int", dest="freq", default=16000)
parser.add_option("-B", "--batch-size", help="Size of batches",
								 action="store", type="int", dest="batchSize", default=20000)
parser.add_option("-b", "--minibatch-size", help="Size of mini-batches",
								 action="store", type="int", dest="miniBatches", default=160)
parser.add_option("-r", "--learning-rate", help="Learning rate",
								 action="store", type="float", dest="learningRate", default=0.01)
parser.add_option("-s", "--skip-seconds", help="Number of seconds to skip at the beginning",
								 action="store", type="float", dest="skipSeconds", default=80)
parser.add_option("-N", "--n-iter", help="Number of iterations to perform",
								 action="store", type="int", dest="nIter", default=80)
parser.add_option("-S", "--stride", help="Stride of the window",
								 action="store", type="int", dest="stride", default=32000)
parser.add_option("-y", "--youtube-id", help="YouTube ID",
								 action="store", type="string", dest="youTubeId", default="XqaJ2Ol5cC4")
parser.add_option("-m", "--mode", help="Mode (train, generate)",
								 action="store", type="string", dest="mode", default="train")

(options, args) = parser.parse_args()

# variables
freq         = options.freq
stride       = options.stride
hiddenUnits  = options.hiddenUnits
batchSize    = options.batchSize
miniBatches  = options.miniBatches
learningRate = options.learningRate
youTubeId    = options.youTubeId
mode         = options.mode
skipSeconds  = options.skipSeconds
nIter        = options.nIter

sequenceSize = batchSize*miniBatches
sequenceSeconds = sequenceSize / freq
idxBegin = skipSeconds * freq
idxEnd   = idxBegin + nIter

# Filenames.
modelBaseName = "lstm-model--id_{0}-batch_{1}-seq_{2}-lr_{3}-nh_{4}".format(youTubeId, batchSize, sequenceSeconds, learningRate, hiddenUnits)
modelFileName = modelBaseName + ".pkl"
graphFileName = modelBaseName + ".png"
soundFileName = modelBaseName + ".wav"

vals = []
error = np.array([0])
minError = np.inf
idx = 0
scaling = 0

# create LSTM
lstm = LSTM(miniBatches, hiddenUnits, miniBatches)

# retrive datastream
print("retrieving data...")
data = YouTubeAudio(youTubeId)
stream = data.get_example_stream()
data_stream = Window(stride, sequenceSize, sequenceSize, True, stream)

# switch to configure training or audio generation
if mode == "train":

	print("training begin...")
	print("Input Size:", batchSize)
	print("minibatches:", miniBatches) 
	print("stride:", stride)
	print("hidden units:", hiddenUnits)
	print("learning rate:", learningRate)
	print("sequence size:", sequenceSize)
	for batch_stream in data_stream.get_epoch_iterator():
		
		# get samples
		u, t = batch_stream
		
		# Start somewhere (after 1 minute)
		if idx >= idxBegin:
			u = np.array(u, dtype=np.float64)
			t = np.array(t, dtype=np.float64)
			# reshape samples into minibatches
			uBatch = np.reshape((u/0x8000), (miniBatches,batchSize)).swapaxes(0,1)
			tBatch = np.reshape((t/0x8000), (miniBatches,batchSize)).swapaxes(0,1)
			print(uBatch)
			# train and find error
			print("train...")
			error  = lstm.train(uBatch, tBatch, learningRate)
			vals.append(error)
			print ("Cost:", error, "at iteration:",idx-(60*freq))			
			
			if error<minError:
				print("LOWEST ERROR")
				minError = error
				f = open(modelFileName, 'wb')
				pickle.dump(lstm.params, f)
				f.close()

		# End somewhere
		if idx >= idxEnd: break
		idx = idx + 1
	
	print("Total sequence trained:", (idx-(80*freq))*(stride/freq), "seconds")
	
	# saving and printing
	plt.plot(vals)
	plt.savefig(graphFileName)			
	f = open(modelFileName, 'wb')
	pickle.dump(vals, f)
	f.close()
	

elif mode == "generate":
	
	# load parameters
	f = open(modelFileName, 'rb') 
	lstm.params = pickle.load(f) #load params from file
	f.close()
	start = 0;
	
	print("generation begin...")
	print("Input Size:", batchSize)
	print("minibatches:", miniBatches) 
	print("Sequence length:", sequenceSize/freq, "s")
	print("stride:", stride)
	print("hidden units:", hiddenUnits)
	print("learning rate:", learningRate)
	for batch_stream in data_stream.get_epoch_iterator():
		
		# Start somewhere
		if idx>idxBegin:
			if start == 0:
				# get samples
				u, t = batch_stream
				u = np.array(u, dtype=np.float64)		
				start = 1
			# reshape samples into minibatches
			uBatch = np.reshape((u/0x8000), (miniBatches,batchSize)).swapaxes(0,1)
			
			# generate 1 batch of data
			prediction = lstm.predict(uBatch)
			#print(prediction)
			prediction = np.reshape(prediction.swapaxes(0,1), u.shape)
			print(prediction, prediction.shape)
			for item in prediction[:stride]:
				vals.append(np.asscalar(item))
				
			# update u
			u = np.append(u[stride:], prediction[:stride], axis=0)
			print ("Iteration:", idx-(60*freq))	
			
		# End somewhere
		if idx>idxEnd:break # iterations
		idx = idx + 1
		
	print("Total sequence size generated:", (idx-(240*freq))*(stride/freq), "seconds")
	
	f = open(modelFileName, 'wb')
	pickle.dump(vals, f)
	f.close()
	plt.plot(vals)
	plt.savefig(graphFileName)
	wave.write(soundFileName, 16000, np.asarray(vals, dtype=np.float64))
			
else:
	print "Error: unknown mode: '{0}'".format(mode)
