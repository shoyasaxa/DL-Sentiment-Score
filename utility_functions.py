import pandas as pd
import numpy as np
import tensorflow as tf
import re
import codecs
import os
from nltk.tokenize import RegexpTokenizer

def load_embeddings(glove_path):
	print("Loading embeddings...")
	weight_vectors = []
	word_idx = {}
	with codecs.open(glove_path, encoding='utf-8') as f:
		for line in f:
			word, vec = line.split(u' ', 1)
			word_idx[word] = len(weight_vectors)
			weight_vectors.append(np.array(vec.split(), dtype=np.float32))
	# Annoying implementation detail; '(' and ')' are replaced by '-LRB-' and
	# '-RRB-' respectively in the parse-trees.
	word_idx[u'-LRB-'] = word_idx.pop(u'(')
	word_idx[u'-RRB-'] = word_idx.pop(u')')
	# Random embedding vector for unknown words.
	weight_vectors.append(np.random.uniform(
	  -0.05, 0.05, weight_vectors[0].shape).astype(np.float32))
	return np.stack(weight_vectors), word_idx

def read_data(data_path):
	# read dictionary into df
	df_data_sentence = pd.read_table(data_path + 'dictionary.txt')
	df_data_sentence_processed = df_data_sentence['Phrase|Index'].str.split('|', expand=True)
	df_data_sentence_processed = df_data_sentence_processed.rename(columns={0: 'Phrase', 1: 'phrase_ids'})

	# read sentiment labels into df
	df_data_sentiment = pd.read_table(data_path + 'sentiment_labels.txt')
	df_data_sentiment_processed = df_data_sentiment['phrase ids|sentiment values'].str.split('|', expand=True)
	df_data_sentiment_processed = df_data_sentiment_processed.rename(columns={0: 'phrase_ids', 1: 'sentiment_values'})


	#combine data frames containing sentence and sentiment
	df_processed_all = df_data_sentence_processed.merge(df_data_sentiment_processed, how='inner', on='phrase_ids')

	return df_processed_all

def split_data(data, split_percent, data_dir):
	msk = np.random.rand(len(data)) < split_percent
	train_only = data[msk]
	test_and_dev = data[~msk]


	msk_test = np.random.rand(len(test_and_dev)) <0.5
	test_only = test_and_dev[msk_test]
	dev_only = test_and_dev[~msk_test]

	# dev_only.to_csv(os.path.join(data_dir, 'TrainingData/dev.csv'))
	# test_only.to_csv(os.path.join(data_dir, 'TrainingData/test.csv'))
	# train_only.to_csv(os.path.join(data_dir, 'TrainingData/train.csv'))

	return train_only.reset_index(), test_only.reset_index(), dev_only.reset_index()

def get_max_seq_len(data):
	total_words = 0
	sequence_length = []
	idx = 0

	for index, row in data.iterrows():

		sentence = (row['Phrase'])
		sentence_words = sentence.split(' ')
		len_sentence_words = len(sentence_words)
		total_words += len_sentence_words

		# get the length of the sequence of each training data
		sequence_length.append(len_sentence_words)

		if idx == 0:
		    max_seq_len = len_sentence_words


		if len_sentence_words > max_seq_len:
		    max_seq_len = len_sentence_words

		# max_seq_len = max(max_seq_len,len_sentence_words)

		idx += 1

	avg_words = total_words/index

	# convert to numpy array
	sequence_length_np = np.asarray(sequence_length)

	return max_seq_len, avg_words, sequence_length_np

def tf_data_pipeline_nltk(data, word_idx, weight_matrix, max_seq_len):

	#training_data = training_data[0:50]

	maxSeqLength = max_seq_len #Maximum length of sentence
	no_rows = len(data)
	ids = np.zeros((no_rows, maxSeqLength), dtype='int32')
	# conver keys in dict to lower case
	word_idx_lwr =  {k.lower(): v for k, v in word_idx.items()}
	idx = 0

	for index, row in data.iterrows():


		sentence = (row['Phrase'])
		#print (sentence)
		tokenizer = RegexpTokenizer(r'\w+')
		sentence_words = tokenizer.tokenize(sentence)
		#print (sentence_words)
		i = 0
		for word in sentence_words:
			#print(index)
			word_lwr = word.lower()
			try:
				#print (word_lwr)
				ids[idx][i] =  word_idx_lwr[word_lwr]

			except Exception as e:
				#print (e)
				#print (word)
				if str(e) == word:
					ids[idx][i] = 0
				continue
			i = i + 1
		idx = idx + 1

	return ids

def get_labels_matrix(data):

	labels = data['sentiment_values']

	lables_float = labels.astype(float)

	cats = ['0','1','2','3','4','5','6','7','8','9']
	labels_mult = (lables_float * 10).astype(int)
	dummies = pd.get_dummies(labels_mult, prefix='', prefix_sep='')
	dummies = dummies.T.reindex(cats).T.fillna(0)
	labels_matrix = dummies.as_matrix()

	return labels_matrix