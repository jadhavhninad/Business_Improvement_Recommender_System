from collections import Counter
from datetime import datetime
import os.path 
import json 
from keras.layers import Embedding, LSTM, Dense, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers import Convolution1D, Flatten
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
 
import numpy as np
import pickle


if not os.path.exists('sentModel2.h5'):
	print('Couldnt find model file...Creating new model...Will take atleast 30 minutes')
	# Load the reviews and parse JSON
	t1 = datetime.now()
	with open("C:\\Users\\rajat\\Desktop\\SML\\Project\\yelp_dataset~\\dataset\\review.json", encoding="utf8") as f:
		#reviews = f.read().strip().split("\n")
		reviews = f.readlines()
	reviews = map(lambda x: x.rstrip(), reviews)
	reviews = [json.loads(review) for review in reviews]
	print(datetime.now() - t1)

	# Get a balanced sample of positive and negative reviews
	texts = [review['text'] for review in reviews]

	# Convert our 5 classes into 2 (negative or positive)
	binstars = [0 if review['stars'] <= 3 else 1 for review in reviews]
	balanced_texts = []
	balanced_labels = []
	limit = 100000  # Change this to grow/shrink the dataset
	neg_pos_counts = [0, 0]
	for i in range(len(texts)):
		polarity = binstars[i]
		if neg_pos_counts[polarity] < limit:
			balanced_texts.append(texts[i])
			balanced_labels.append(binstars[i])
			neg_pos_counts[polarity] += 1

			
	Counter(balanced_labels)
	if  not os.path.exists('tokenizer.pickle'):	
		tokenizer = Tokenizer(num_words=20000)
		tokenizer.fit_on_texts(balanced_texts)
	else:
		print('Tokenizer found no need to read reviews')
		with open('tokenizer.pickle', 'rb') as handle:
			tokenizer = pickle.load(handle)
	sequences = tokenizer.texts_to_sequences(balanced_texts)
	data = pad_sequences(sequences, maxlen=300)

	'''
	model = Sequential()
	model.add(Embedding(20000, 128, input_length=300))
	model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	'''
	model = Sequential()
	model.add(Embedding(20000, 128, input_length=300))
	model.add(Conv1D(64, 3, padding="same"))
	model.add(Conv1D(32, 3, padding="same"))
	model.add(Conv1D(16, 3, padding="same"))
	model.add(Flatten())
	model.add(Dropout(0.2))
	model.add(Dense(180,activation='sigmoid'))
	model.add(Dropout(0.2))
	model.add(Dense(1,activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])	

	#model.fit(data, np.array(balanced_labels), validation_split=0.5, epochs=3)
	model.fit(data, np.array(balanced_labels), epochs=3, batch_size=64)
	
	#Saving model
	with open('tokenizer.pickle', 'wb') as handle:
		pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
	model.save('sentModel2.h5')
else:
	print('Model already built')
	model3 = load_model('sentModel2.h5')
	model2 = load_model('sentModel1.h5')
	model1 = load_model('sentModel.h5')
	
	texts = ['We came in this afternoon and I ordered a dirty chai and it was delish! The shop had a lot of awesome things for sale like some really beautiful handmade jewelry, cigars, shirts, coffee mugs, etc. I just booked my graduation party here and can not wait, the ceremony area outside is beautiful! Highly recommended', 'Nice store , horrible customer service. We decided to goto Shane & co. To have a custom wedding ring made in Aug 2017. As the commercial says they can do custom rings.We were greeted by a young male, as soon as we walked in the front door he immediately looked at the jewlery I was wearing, I thought to myself seeing him do this. I new exactly how this was going to go. I then explained and brought a sketch of what we were looking for : marquis cut center stone with a  channel setting in baguettes double bypass ring 14k yellow gold. I did mention that we have just celebrated our 20 year wedding anniversary, and this is what we wanted. He then said have we been saving for 20 years for this ring. I got extremely insulted. I said dont worry about the money. It was not a problem. He then finally said I think I know what you are looking for,  my grandmother had that ring. Another insult, really? I said if you can not help us let us know and we will go elsewhere. So after 45 minutes of wasting our time, we left extremely unhappy with their customer service skills. We are getting the ring made by another jewler, who was more then happy to help us.', 'Ambience was okay', 'Not good', 'Pretty good', 'Worst ever', 'Worst', 'Best ever', 'Good', 'Awesome']
	
	# loading
	with open('tokenizer.pickle', 'rb') as handle:
		tokenizer = pickle.load(handle)
	print('Successfully loaded my awesome pickle')
	sequences = tokenizer.texts_to_sequences(texts)
	data = pad_sequences(sequences, maxlen=300)
	print(list(map(lambda x: x[:10], texts)))
	print( list( map(lambda x: 'Pos' if x>0.5 else 'Neg', model1.predict(data) ) ) )
	print( list( map(lambda x: 'Pos' if x>0.5 else 'Neg', model2.predict(data) ) ) )
	print( list( map(lambda x: 'Pos' if x>0.5 else 'Neg', model3.predict(data) ) ) )

	
	from sklearn.externals import joblib
	 
	vectorizer = joblib.load("tfidf_vectorizer.pickle")
	classifier = joblib.load("svm_classifier.pickle")
	 
	# note that we should call "transform" here instead of the "fit_transform" from earlier
	Xs = vectorizer.transform(texts)
	 
	# get predictions for each of your new texts
	predictions = classifier.predict(Xs)
	print( list( map(lambda x: 'Pos' if x>0.5 else 'Neg', predictions ) ) )
	