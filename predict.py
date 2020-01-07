import utility_functions as uf 
import os
import numpy as np
import sys
import h5py
from keras.models import load_model,model_from_json
from nltk.tokenize import RegexpTokenizer
import pandas as pd 
import numpy as np 


import re

alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr|Prof)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"

def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences

def predict_score(trained_model, sentence, word_idx):
	print(sentence)
	sentence_list = []
	sentence_list_np = np.zeros((56,1))
	# split the sentence into its words and remove any punctuations.
	tokenizer = RegexpTokenizer(r'\w+')
	data_sample_list = tokenizer.tokenize(sentence)

	labels = np.array([1,2,3,4,5,6,7,8,9,10], dtype = "int")
	#word_idx['I']
	# get index for the live stage
	data_index = np.array([word_idx[word.lower()] if word.lower() in word_idx else 0 for word in data_sample_list])
	data_index_np = np.array(data_index)
	print(data_index_np)

	# padded with zeros of length 56 i.e maximum length
	padded_array = np.zeros(56) # use the def maxSeqLen(training_data) function to detemine the padding length for your data
	padded_array[:data_index_np.shape[0]] = data_index_np
	data_index_np_pad = padded_array.astype(int)
	sentence_list.append(data_index_np_pad)
	sentence_list_np = np.asarray(sentence_list)

	# get score from the model
	score = trained_model.predict(sentence_list_np, batch_size=1, verbose=0)
	#print (score)

	# single_score = np.round(np.argmax(score)/10, decimals=2) # maximum of the array i.e single band

	# weighted score of top 3 bands
	top_3_index = np.argsort(score)[0][-3:]
	top_3_scores = score[0][top_3_index]
	top_3_weights = top_3_scores/np.sum(top_3_scores)
	single_score_dot = np.round(np.dot(top_3_index, top_3_weights)/10, decimals = 2)


	new_range = 4 
	new_min = 1 
	old_range = 2 

	scaled_score = round((((single_score_dot + new_min) * new_range)/old_range) + new_min,2)

	#print (single_score)
	return scaled_score

def test_predict(path):
	glove_file = path+'/Data/glove/glove_6B_100d.txt'
	weight_matrix, word_idx = uf.load_embeddings(glove_file)
	# weight_path = path +'/model/best_weights_bi_glove.hdf5'

	# models = [] 

	# i=0
	# for model_weight_file in os.listdir(path+'/model/keras_tuner'):
	# 	json_file = open(root_path+"/keras_tuner/models/best_model_{}.json".format(i))
	# 	loaded_model_json = json_file.read()
	# 	json_file.close()
	# 	loaded_model = model_from_json(loaded_model_json)
	# 	loaded_model.load_weights(root_path+"/keras_tuner/weights/best_model_{}.h5".format(i))
	# 	loaded_model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
	# 	models.append(loaded_model)
	# 	i+=1

	weight_path = path +'/model/best_model.hdf5'
	loaded_model = load_model(weight_path)
	loaded_model.summary()
	# data_sample = "Great!! it is raining today!!"

	sentences = [
		"Amazing Professor and the best class I have ever had at Columbia. He is so caring and such an inspirational speaker. I would definitely recommend!",
		"He was an okay professor. Very normal workload, not a bad class",
		"Decent class, decent workload, okay final. Overall, I would recommend",
		"Workload was pretty rough, but overall, Benajiba is not a bad choice for NLP",
		"I expected more from this class. The professor will publicly call you out in front of the class if you get a problem wrong. I would avoid it",
		"This class will teach you a lot about proofs. If you like hard math, then I would recommend. If you don't like complicated equations, then I wouldn't recommend it.",
		"This class will teach you a lot about proofs.",
		"If you like hard math, then I would recommend",
		"If you don't like complicated equations, then I wouldn't recommend it.",
		"Really bad professor. He mumbles and is very unclear. Very high expectations on the exam that is almost impossible to meet. Absolutely avoid this professor at all costs.",
		"avoid at all costs",
		"he is really bad",
		"don't ever take this course",
		"why would the university keep this professor around?",
		"he should be fired immediately",
		"no one should ever take this course",
		"why would anyone ever want to take this course",
		"run away from him",
		"Extremely challenging course for no reason. Highly suggest not taking this course"
	]

	for sentence in sentences:
		print(predict_score(loaded_model,sentence,word_idx))

		# for i, model in enumerate(models):
		# 	print("model {}:".format(i))
		# 	print(predict_score(model,sentence,word_idx))
   
def predict_review_score(path,data_path):
	glove_file = path+'/Data/glove/glove_6B_100d.txt'
	weight_matrix, word_idx = uf.load_embeddings(glove_file)
	weight_path = path +'/model/best_model.hdf5'
	loaded_model = load_model(weight_path)

	df = pd.read_excel(data_path)
	df = df.loc[df["is_workload"] == 0].copy()

	df['review'] = df['review'].replace('\n','. ', regex=True)
	reviews = df["review"].values.tolist()
	is_workload_list = df["is_workload"].values.tolist()
	prof_ids = df["prof_id_culpa"].values.tolist() 
	review_ids = df["id_culpa"].values.tolist()

	review_scores = [] 
	for i, review in enumerate(reviews):
		score_sum = 0
		# sentences = reviews.split('.')
		sentences = split_into_sentences(reviews)
		for sentence in sentences: 
			words = sentence.split(' ')
			if len(words) > 55:
				sentence = " ".join(words[:55]) 
			
			score = predict_score(loaded_model,sentence,word_idx)

			print(sentence)
			print(score)

			score_sum += score 
		review_scores.append(score_sum/len(sentences))

	df["scores"] = review_scores

	new_range = 4 
	new_min = 1 
	old_range = 2 
	df["scores"] = (((df["scores"] + 1) * new_range)/old_range)+new_min

	df[['prof_id_culpa','id_culpa','review','scores']].to_excel("/output/dl_review_scores_v1.xlsx")

	df_prof_scores = df.groupby(['prof_id_culpa']).mean()
	df_prof_scores.to_excel('/output/dl_prof_scores_v1.xlsx')


if __name__ == "__main__":
	test_predict(sys.argv[1])
	predict_review_score(sys.argv[1],sys.argv[2])