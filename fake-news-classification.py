import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
import math
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation,TimeDistributed
from keras.layers import Bidirectional, LSTM
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate,BatchNormalization,MaxPooling1D, Convolution1D
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D, RepeatVector, Permute, merge
from keras import backend as K
from keras import layers
from keras.models import Sequential, models
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from math import sqrt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from utils import *
from attack_utils import *




class CNNClassifier():

	def __init__(self, tokenizer,embedding_matrix,MAX_SEQUENCE_LENGTH=5000, MAX_NUM_WORDS=25000, EMBEDDING_DIM=300):
		'''
			MAX_SEQUENCE_LENGTH = 5000
			MAX_NUM_WORDS = 25000
			EMBEDDING_DIM = 300
		'''
		self.type = "CNN"
		self.MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH
		self.MAX_NUM_WORDS = MAX_NUM_WORDS
		self.EMBEDDING_DIM = EMBEDDING_DIM
		self.model = self.createModel()
		self.history = None
		self.tokenizer = tokenizer
		self.embedding_matrix = embedding_matrix

	def createModel(self):
		dropout_prob = [0.2,0.2]
		hidden_dims = 50
		filter_sizes  = (3,8)
		num_filters = 10
		BATCH_SIZE = 32

		input_shape = (MAX_SEQUENCE_LENGTH,)
		model_input = Input(shape=input_shape)
		z = Embedding(MAX_NUM_WORDS,
		              EMBEDDING_DIM,
		              weights=[self.embedding_matrix],
		              input_length=MAX_SEQUENCE_LENGTH)(model_input)

		conv_blocks = []
		for sz in filter_sizes:
		    conv = Convolution1D(filters=num_filters,
		                         kernel_size=sz,
		                         padding="valid",
		                         activation="relu",
		                         strides=1)(z)
		    conv = MaxPooling1D(pool_size=2)(conv)
		    conv = Flatten()(conv)
		    conv_blocks.append(conv)
		z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

		z = Dropout(dropout_prob[1])(z)
		z = Dense(hidden_dims, activation="relu")(z)
		model_output = Dense(1, activation="sigmoid")(z)

		model = Model(model_input, model_output)
		model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy",f1])
		return model

	def trainModel(self,SAVE_MODEL_PATH,initial_epoch_num=0):
		
	    # training
	    TRAINING_BATCH_SIZE      = 64
	    # TRAINING_SHUFFLE_BUFFER  = 5000
	    # TRAINING_BN_MOMENTUM     = 0.99
	    # TRAINING_BN_EPSILON      = 0.001
	    TRAINING_LR_MAX          = 0.001
	    # TRAINING_LR_SCALE        = 0.1
	    # TRAINING_LR_EPOCHS       = 2
	    TRAINING_LR_INIT_SCALE   = 0.01
	    TRAINING_LR_INIT_EPOCHS  = 3
	    TRAINING_LR_FINAL_SCALE  = 0.01
	    TRAINING_LR_FINAL_EPOCHS = 7

	    # training (derived)
	    TRAINING_NUM_EPOCHS = TRAINING_LR_INIT_EPOCHS + TRAINING_LR_FINAL_EPOCHS
	    TRAINING_LR_INIT    = TRAINING_LR_MAX*TRAINING_LR_INIT_SCALE
	    TRAINING_LR_FINAL   = TRAINING_LR_MAX*TRAINING_LR_FINAL_SCALE

		def lr_schedule(epoch):    
		    # staircase
		    # lr = TRAINING_LR_MAX*math.pow(TRAINING_LR_SCALE, math.floor(epoch/TRAINING_LR_EPOCHS))
		    # linear warmup followed by cosine decay
		    if epoch < TRAINING_LR_INIT_EPOCHS:
		        lr = (TRAINING_LR_MAX - TRAINING_LR_INIT)*(float(epoch)/TRAINING_LR_INIT_EPOCHS) + TRAINING_LR_INIT
		    else:
		        lr = ((TRAINING_LR_MAX - TRAINING_LR_FINAL)*
		              max(0.0, math.cos(((float(epoch) - TRAINING_LR_INIT_EPOCHS)/
		                                 (TRAINING_LR_FINAL_EPOCHS - 1.0))*(math.pi/2.0))) + 
		              TRAINING_LR_FINAL)
		    # debug - learning rate display
		    # print(epoch)
		    # print(lr)
		    return lr

		# train the model
		callbacks = [keras.callbacks.LearningRateScheduler(lr_schedule),
		             keras.callbacks.ModelCheckpoint(filepath=SAVE_MODEL_PATH+'model_{epoch}.h5', 
		                                             save_best_only=True, monitor='val_loss', verbose=1)]
		# initial_epoch_num = 0
		# example of restarting training after a crash from the last saved checkpoint
		if not initial_epoch==0:
			self.model.load_weights(SAVE_MODEL_PATH+self.type+'_model_'+str(initial_epoch)+'.h5') # replace X with the last saved checkpoint number
		# initial_epoch_num = 5                            # replace X with the last saved checkpoint number
		self.history = model.fit(x_train, y_train,
			shuffle=True,batch_size=TRAINING_BATCH_SIZE,
          	epochs=TRAINING_NUM_EPOCHS, 
          	verbose=1, 
          	callbacks=callbacks, 
          	validation_data=(x_val, y_val), 
          	initial_epoch=initial_epoch_num)

	# evaluate model
	def predict(self,X):
    	return np.rint(self.model.predict(X)) # threshold the predictions to retrieve labels

def run_CNN_model_creation():
	MAX_SEQUENCE_LENGTH = 5000
	MAX_NUM_WORDS = 25000
	EMBEDDING_DIM = 300
	TEST_SPLIT = 0.3
	EMBEDDING_FILE=f'glove.6B.{EMBEDDING_DIM}d.txt'
	TEXT_DATA = 'data/fake_or_real_news.csv'

	# saving model
	SAVE_MODEL_PATH = './save/model/'
	!mkdir -p "$SAVE_MODEL_PATH"

	df = read_csv(TEXT_DATA=TEXT_DATA):
	texts, labels = getTextLabels(df=df)
	tokenizer, embedding_matrix = getEmbedding(texts=texts,EMBEDDING_FILE=EMBEDDING_FILE, EMBEDDING_DIM=EMBEDDING_DIM, 
		MAX_NUM_WORDS=MAX_NUM_WORDS)
	x_train, x_val, y_train, y_val = split_data(data=data,labels=labels, TEST_SPLIT=TEST_SPLIT)

	cnn = CNNClassifier(tokenizer=tokenizer,embedding_matrix=embedding_matrix,
		MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH, MAX_NUM_WORDS=MAX_NUM_WORDS, 
		EMBEDDING_DIM=EMBEDDING_DIM)
	print(cnn.model.summary())
	cnn.trainModel(SAVE_MODEL_PATH=SAVE_MODEL_PATH,initial_epoch=0)

	# Plot training & validation accuracy values
	# print(history.history)
	plt.plot(cnn.history.history['accuracy'])
	plt.plot(cnn.history.history['val_accuracy'])
	plt.title('Model accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.show()

	train_acc, test_acc = evaluate_model(cnn.predict,x_train, y_train, x_val, y_val)
	print("Training Accuracy: {:.2f}%".format(train_acc*100))
	print("Testing Accuracy: {:.2f}%".format(test_acc*100))
	return cnn




class LSTMClassifier():

	def __init__(self, tokenizer,embedding_matrix,MAX_SEQUENCE_LENGTH=5000, MAX_NUM_WORDS=25000, EMBEDDING_DIM=300):
		'''
			MAX_SEQUENCE_LENGTH = 5000
			MAX_NUM_WORDS = 25000
			EMBEDDING_DIM = 300
		'''
		self.type = "LSTM"
		self.MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH
		self.MAX_NUM_WORDS = MAX_NUM_WORDS
		self.EMBEDDING_DIM = EMBEDDING_DIM
		self.model = self.createModel()
		self.history = None
		self.tokenizer = tokenizer
		self.embedding_matrix = embedding_matrix

	def createModel(self):
		BATCH_SIZE = 128

		# convs = []
		# filter_sizes = [3,4,5]
		model=None
		sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
		embedding_layer = Embedding(MAX_NUM_WORDS,
		                            EMBEDDING_DIM,
		                            weights=[embedding_matrix],
		                            input_length=MAX_SEQUENCE_LENGTH,
		                            trainable=False)


		embedded_sequences = embedding_layer(sequence_input)
		# l_lstm = Bidirectional(LSTM(100, return_sequences=True))(embedded_sequences)
		l_lstm = LSTM(100, return_sequences=True)(embedded_sequences)
		f = Flatten()(l_lstm)
		# preds = Dense(1, activation='softmax')(f)
		preds = Dense(1, activation='sigmoid')(f)
		# preds = TimeDistributed(2, Dense(200,activation='relu'))(l_gru)
		model = Model(sequence_input, preds)

		model.compile(loss='binary_crossentropy',
		              optimizer='rmsprop',
		              metrics=['acc'])

		print("model fitting - LSTM network")
		return model

	def trainModel(self,SAVE_MODEL_PATH,initial_epoch_num=0):
		
	    # training
	    TRAINING_BATCH_SIZE      = 64
	    # TRAINING_SHUFFLE_BUFFER  = 5000
	    # TRAINING_BN_MOMENTUM     = 0.99
	    # TRAINING_BN_EPSILON      = 0.001
	    TRAINING_LR_MAX          = 0.001
	    # TRAINING_LR_SCALE        = 0.1
	    # TRAINING_LR_EPOCHS       = 2
	    TRAINING_LR_INIT_SCALE   = 0.01
	    TRAINING_LR_INIT_EPOCHS  = 3
	    TRAINING_LR_FINAL_SCALE  = 0.01
	    TRAINING_LR_FINAL_EPOCHS = 7

	    # training (derived)
	    TRAINING_NUM_EPOCHS = TRAINING_LR_INIT_EPOCHS + TRAINING_LR_FINAL_EPOCHS
	    TRAINING_LR_INIT    = TRAINING_LR_MAX*TRAINING_LR_INIT_SCALE
	    TRAINING_LR_FINAL   = TRAINING_LR_MAX*TRAINING_LR_FINAL_SCALE

		def lr_schedule(epoch):    
		    # staircase
		    # lr = TRAINING_LR_MAX*math.pow(TRAINING_LR_SCALE, math.floor(epoch/TRAINING_LR_EPOCHS))
		    # linear warmup followed by cosine decay
		    if epoch < TRAINING_LR_INIT_EPOCHS:
		        lr = (TRAINING_LR_MAX - TRAINING_LR_INIT)*(float(epoch)/TRAINING_LR_INIT_EPOCHS) + TRAINING_LR_INIT
		    else:
		        lr = ((TRAINING_LR_MAX - TRAINING_LR_FINAL)*
		              max(0.0, math.cos(((float(epoch) - TRAINING_LR_INIT_EPOCHS)/
		                                 (TRAINING_LR_FINAL_EPOCHS - 1.0))*(math.pi/2.0))) + 
		              TRAINING_LR_FINAL)
		    # debug - learning rate display
		    # print(epoch)
		    # print(lr)
		    return lr

		# train the model
		callbacks = [keras.callbacks.LearningRateScheduler(lr_schedule),
		             keras.callbacks.ModelCheckpoint(filepath=SAVE_MODEL_PATH+'model_{epoch}.h5', 
		                                             save_best_only=True, monitor='val_loss', verbose=1)]
		# initial_epoch_num = 0
		# example of restarting training after a crash from the last saved checkpoint
		if not initial_epoch==0:
			self.model.load_weights(SAVE_MODEL_PATH+self.type+'_model_'+str(initial_epoch)+'.h5') # replace X with the last saved checkpoint number
		# initial_epoch_num = 5                            # replace X with the last saved checkpoint number
		self.history = model.fit(x_train, y_train,
			shuffle=True,batch_size=TRAINING_BATCH_SIZE,
          	epochs=TRAINING_NUM_EPOCHS, 
          	verbose=1, 
          	callbacks=callbacks, 
          	validation_data=(x_val, y_val), 
          	initial_epoch=initial_epoch_num)

	# evaluate model
	def predict(self,X):
    	return np.rint(self.model.predict(X)) # threshold the predictions to retrieve labels


def run_LSTM_model_creation():
	MAX_SEQUENCE_LENGTH = 5000
	MAX_NUM_WORDS = 25000
	EMBEDDING_DIM = 300
	TEST_SPLIT = 0.3
	EMBEDDING_FILE=f'glove.6B.{EMBEDDING_DIM}d.txt'
	TEXT_DATA = 'data/fake_or_real_news.csv'

	# saving model
	SAVE_MODEL_PATH = './save/model/'
	!mkdir -p "$SAVE_MODEL_PATH"

	df = read_csv(TEXT_DATA=TEXT_DATA):
	texts, labels = getTextLabels(df=df)
	tokenizer, embedding_matrix = getEmbedding(texts=texts,EMBEDDING_FILE=EMBEDDING_FILE, EMBEDDING_DIM=EMBEDDING_DIM, 
		MAX_NUM_WORDS=MAX_NUM_WORDS)
	x_train, x_val, y_train, y_val = split_data(data=data,labels=labels, TEST_SPLIT=TEST_SPLIT)

	lstm = LSTMClassifier(tokenizer=tokenizer,embedding_matrix=embedding_matrix,
		MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH, MAX_NUM_WORDS=MAX_NUM_WORDS, 
		EMBEDDING_DIM=EMBEDDING_DIM)
	print(lstm.model.summary())
	lstm.trainModel(SAVE_MODEL_PATH=SAVE_MODEL_PATH,initial_epoch=0)

	# Plot training & validation accuracy values
	# print(history.history)
	plt.plot(lstm.history.history['accuracy'])
	plt.plot(lstm.history.history['val_accuracy'])
	plt.title('Model accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.show()

	train_acc, test_acc = evaluate_model(lstm.predict,x_train, y_train, x_val, y_val)
	print("Training Accuracy: {:.2f}%".format(train_acc*100))
	print("Testing Accuracy: {:.2f}%".format(test_acc*100))
	return lstm



def attack(X,model,SAVE_MODEL_PATH = './save/model/'):
	'''
	replace X with the last saved checkpoint number
	model CNN or LSTM whichever used
	'''
	logFile = SAVE_MODEL_PATH+model.type+'_log.csv'
	model.model.load_weights(SAVE_MODEL_PATH+model.type+'_model_'+X+'.h5') 
	fake_texts = list(df['label'].apply(lambda x: x == 'FAKE'))
	fake_texts = df[fake_texts]
	fake_texts = fake_texts['text']

	fake_text_vectors = model.tokenizer.texts_to_sequences(fake_texts)
	fake_text_vectors = pad_sequences(fake_text_vectors, 
                     maxlen=model.MAX_SEQUENCE_LENGTH, 
                     padding='pre', 
                     truncating='pre')
	correct_clasified_fakes = evaluate_ourfakes(model, fake_text_vectors)
	correct_fakes =tokenizer.sequences_to_texts(correct_clasified_fakes)
	for i in tqdm(range(200)):
    with open(,mode='a') as f:
      # success_on[i],_ = attack(correct_fakes[i],1)
      pertube, sent = attack(correct_fakes[i],1)
      if not sent == None:
        f.write(str(pertube)+','+sent+'\n')
