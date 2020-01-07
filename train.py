import utility_functions as uf 

import os
import numpy as np
import sys

import keras
from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense, Flatten, LSTM, Bidirectional, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import h5py
# from kerastuner.tuners import RandomSearch
# # from kerastuner import HyperParameters
# from kerastuner.engine.hypermodel import HyperModel
# from kerastuner.engine.hyperparameters import HyperParameters

from nltk.tokenize import RegexpTokenizer

def load_all_data(data_dir,prediction_path, glove_file, first_run):
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


# class MyHyperModel(HyperModel):
# 	def __init__(self, weight_matrix, max_words, embedding_dim):
# 		self.weight_matrix = weight_matrix,
# 		self.max_words = max_words,
# 		self.embedding_dim = embedding_dim

# 	def build(self,hp):
# 		model = Sequential()

# 		# e = Embedding(len(self.weight_matrix), self.embedding_dim, weights=[self.weight_matrix], input_length=self.max_words, trainable=False)
# 		model.add(Embedding(len(self.weight_matrix), self.embedding_dim, weights=[self.weight_matrix], trainable=False))

# 		for i in range(hp.Int('num_layers_lstm',1,4)):
# 			model.add(Bidirectional(LSTM(
# 				units=hp.Int('units_lstm',min_value=64,max_value=1024,step=64), 
# 				dropout=hp.Float('lstm_dropout',min_value=0.0,max_value=0.5,step=0.1), 
# 				recurrent_dropout=hp.Float('lstm_recurr_dropout',min_value=0.0,max_value=0.5,step=0.1)
# 			)))

# 		for i in range(hp.Int('num_layers_dense',1,6)):
# 			model.add(Dense(
# 				units=hp.Int('units_dense',min_value=256,max_value=2048,step=256), 
# 				activation='relu'
# 			))
# 			model.add(Dropout(
# 				rate=hp.Float('dense_dropout',min_value=0.0,max_value=0.5,step=0.1)
# 			))

# 		model.add(Dense(10, activation='softmax'))
# 		# try using different optimizers and different optimizer configs
# 		model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
# 		print(model.summary())

# 		return model 


def train(root_path):
	BATCH_SIZE = 1024
	EMBEDDING_DIM = 100 

	data_directory = root_path + '/Data'
	prediction_path = root_path + '/Data/output/test_pred.csv'
	glove_file = root_path + '/Data/glove/glove_6B_100d.txt'

	first_run = True 

	train_x, train_y, test_x, test_y, val_x, val_y, weight_matrix, word_idx, max_seq_length = load_all_data(data_directory,prediction_path, glove_file, first_run)

	model = build_model(weight_matrix, max_seq_length, EMBEDDING_DIM)

	saveBestModel = keras.callbacks.ModelCheckpoint(root_path+'/model/best_model.hdf5', monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
	earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')

	# print("Initializing RandomSearch Tuner...")

	# tuner = RandomSearch(
	# 	 MyHyperModel(weight_matrix=weight_matrix, max_words=max_seq_length, embedding_dim=EMBEDDING_DIM),
	# 	 objective='val_accuracy',
	# 	 max_trials=3,
	# 	 executions_per_trial=1 
	# )

	# print("Searching...")
	# tuner.search(train_x, train_y, batch_size=BATCH_SIZE, epochs=15,validation_data=(val_x, val_y))
	# tuner.search(train_x, train_y, batch_size=BATCH_SIZE, epochs=25,validation_data=(val_x, val_y), callbacks=[saveBestModel, earlyStopping])

	# best_models = tuner.get_best_models(num_models=2)
	# for i, model in enumerate(best_models):
	# 	score, acc = model.evaluate(test_x, test_y, batch_size=BATCH_SIZE)
	# 	print("model {}:".format(i+1))
	# 	print('Test score:', score)
	# 	print('Test accuracy:', acc)
	# 	model.save_weights(root_path+"/keras_tuner/weights/best_model_{}.h5".format(i))
	# 	model_json = model.to_json()
	# 	with open(root_path+"/keras_tuner/models/best_model_{}.json".format(i),"w") as json_file:
	# 		json_file.write(model_json)
	# 	print("model {} saved".format(i+1))


	# Fit the model
	model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=25,validation_data=(val_x, val_y), callbacks=[saveBestModel, earlyStopping])
	# Final evaluation of the model
	score, acc = model.evaluate(test_x, test_y, batch_size=BATCH_SIZE)

	print('Test score:', score)
	print('Test accuracy:', acc)

	model.save_weights(root_path+"/model/best_model.h5")
	print("Saved model to disk")

if __name__ == "__main__":
	train(sys.argv[1])
