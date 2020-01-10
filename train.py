import utility_functions as uf 

import os
import numpy as np
import sys

import keras
from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense, Flatten, LSTM, Bidirectional, Dropout, Conv1D, MaxPooling1D, Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import h5py
import matplotlib.pyplot as plt 
# from kerastuner.tuners import RandomSearch
# # from kerastuner import HyperParameters
# from kerastuner.engine.hypermodel import HyperModel
# from kerastuner.engine.hyperparameters import HyperParameters

from nltk.tokenize import RegexpTokenizer

def load_all_data(data_dir,prediction_path, glove_file):
	weight_matrix, word_idx = uf.load_embeddings(glove_file)

	data = uf.read_data(data_dir+'/')
	train_data, test_data, dev_data = uf.split_data(data, 0.8, data_dir)

	max_seq_length, avg_words, sequence_length = uf.get_max_seq_len(data)

	train_x = uf.tf_data_pipeline_nltk(train_data, word_idx, weight_matrix, max_seq_length)
	test_x = uf.tf_data_pipeline_nltk(test_data, word_idx, weight_matrix, max_seq_length)
	val_x = uf.tf_data_pipeline_nltk(dev_data, word_idx, weight_matrix, max_seq_length)

	train_y = uf.get_labels_matrix(train_data)
	val_y = uf.get_labels_matrix(dev_data)
	test_y = uf.get_labels_matrix(test_data)

	print("Training data: ")
	print(train_x.shape)
	print(train_y.shape)

	# Summarize number of classes
	print("Classes: ")
	print(np.unique(train_y.shape[1]))

	return train_x, train_y, test_x, test_y, val_x, val_y, weight_matrix, word_idx, max_seq_length

def build_model(weight_matrix, max_words, EMBEDDING_DIM):
	model = Sequential()
	model.add(Embedding(len(weight_matrix), EMBEDDING_DIM, weights=[weight_matrix], input_length=max_words, trainable=False))
	model.add(Bidirectional(LSTM(256, dropout=0.2, recurrent_dropout=0.2)))
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.50))
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.50))
	model.add(Dense(10, activation='softmax'))
	# try using different optimizers and different optimizer configs
	model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
	print(model.summary())

	return model

def build_model_big_lstm(weight_matrix, max_words, EMBEDDING_DIM):
	model = Sequential()
	model.add(Embedding(len(weight_matrix), EMBEDDING_DIM, weights=[weight_matrix], input_length=max_words, trainable=False))
	model.add(Bidirectional(LSTM(256, dropout=0.4, recurrent_dropout=0.4,return_sequences=True)))
	model.add(Bidirectional(LSTM(128, dropout=0.5, recurrent_dropout=0.5)))
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.50))
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.50))
	model.add(Dense(10, activation='softmax'))
	# try using different optimizers and different optimizer configs
	model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
	print(model.summary())

	return model

def build_model_cnn_lstm(weight_matrix, max_words, EMBEDDING_DIM):
	model = Sequential()
	model.add(Embedding(len(weight_matrix), EMBEDDING_DIM, weights=[weight_matrix], input_length=max_words, trainable=False))
	model.add(Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'))
	model.add(MaxPooling1D(pool_size=4))
	model.add(Bidirectional(LSTM(512, dropout=0.2, recurrent_dropout=0.2)))
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.50))
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.50))
	model.add(Dense(10, activation='softmax'))
	# try using different optimizers and different optimizer configs
	model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
	print(model.summary())

	return model

def build_model_cnn(weight_matrix, max_words, EMBEDDING_DIM):
	model = Sequential()
	model.add(Embedding(len(weight_matrix), EMBEDDING_DIM, weights=[weight_matrix], input_length=max_words, trainable=False))
	model.add(Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'))
	model.add(MaxPooling1D(pool_size=4))
	model.add(Conv1D(filters=32, kernel_size=3,padding='same',activation='relu'))
	model.add(MaxPooling1D(pool_size=2))

	model.add(Flatten())

	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.50))
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.50))
	model.add(Dense(10, activation='softmax'))
	# try using different optimizers and different optimizer configs
	model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
	print(model.summary())

	return model


def train(root_path):
	BATCH_SIZE = 1024
	# EMBEDDING_DIM = 100 
	EMBEDDING_DIM = 200
	data_directory = root_path + '/Data'
	prediction_path = root_path + '/Data/output/test_pred.csv'
	glove_file =  root_path + '/Data/glove/glove_6B_100d.txt'
	glove_file_twitter = root_path + '/Data/glove/glove.twitter.27B.200d.txt'

	train_x, train_y, test_x, test_y, val_x, val_y, weight_matrix, word_idx, max_seq_length = load_all_data(data_directory,prediction_path, glove_file_twitter)

	model = build_model_big_lstm(weight_matrix, max_seq_length, EMBEDDING_DIM)

	saveBestModel = keras.callbacks.ModelCheckpoint(root_path+'/model/best_model.hdf5', monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
	earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')

	# Fit the model
	history = model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=20,validation_data=(val_x, val_y), callbacks=[saveBestModel, earlyStopping])

	try: 
		plt.plot(history.history['acc'])
		plt.plot(history.history['val_acc'])
		plt.title('Model accuracy')
		plt.ylabel('Accuracy')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Val'], loc='upper left')
		plt.show()
	except Exception as e:
		print(e)

	# Final evaluation of the model
	score, acc = model.evaluate(test_x, test_y, batch_size=BATCH_SIZE)

	print('Test score:', score)
	print('Test accuracy:', acc)

	model.save_weights(root_path+"/model/best_model.h5")
	print("Saved model to disk")

if __name__ == "__main__":
	train(sys.argv[1])
