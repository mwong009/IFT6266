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

# variables
freq = 16000
stride = 32000
hiddenUnits = 250
batchSize = 20000
miniBatches = 160
sequenceSize = batchSize*miniBatches
learningRate = 0.01 # learning rate
vals = []
error = np.array([0])
minError = np.inf
idx = 0
scaling = 0

# create LSTM
lstm = LSTM(miniBatches, hiddenUnits, miniBatches)

# retrive datastream
print("retrieving data...")
data = YouTubeAudio('XqaJ2Ol5cC4')
stream = data.get_example_stream()
data_stream = Window(stride, sequenceSize, sequenceSize, True, stream)

# switch to configure training or audio generation
training = True

if training == True:

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
		if idx>(80*freq):
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
				f = open('LSTM_MODEL_BATCH20000_200s_LR001_HU250.pkl', 'wb')
				pickle.dump(lstm.params, f)
				f.close()

		# End somewhere
		if idx>(80*freq+160): break # 160 iterations
		idx = idx + 1
	
	print("Total sequence trained:", (idx-(80*freq))*(stride/freq), "seconds")
	
	# saving and printing
	plt.plot(vals)
	plt.savefig('LSTM_PLOT_BATCH20000_200s_LR001_HU250.png')			
	f = open('LSTM_PLOT_BATCH20000_200s_LR001_HU250.pkl', 'wb')
	pickle.dump(vals, f)
	f.close()
	

if training == False:
	
	# load parameters
	f = open('LSTM_MODEL_BATCH20000_200s_LR001_HU250.pkl', 'rb') 
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
		if idx>(240*freq):
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
		if idx>(240*freq+40):break # iterations
		idx = idx + 1
		
	print("Total sequence size generated:", (idx-(240*freq))*(stride/freq), "seconds")
	
	f = open('LSTM_GEN_BATCH20000_200s_LR005_HU250.pkl', 'wb')
	pickle.dump(vals, f)
	f.close()
	plt.plot(vals)
	plt.savefig('LSTM_GEN_PLOT_BATCH20000_200s_LR005_HU250.png')
	wave.write('LSTM_GEN_PLOT_BATCH20000_200s_LR005_HU250.wav', 16000, np.asarray(vals, dtype=np.float64))
			

