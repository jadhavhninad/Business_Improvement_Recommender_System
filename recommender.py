import operator
import logging
import gensim
import pickle
import time
import nltk
import re
import numpy as np
from pymongo import MongoClient
from settings import Settings
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences


class PreProcessing:

    def __init__(self,IDS):

        self.businessCollection = MongoClient(Settings.MONGO_CONNECTION_STRING)[Settings.RAW_DATABASE][Settings.YELP_RAW_BUSINESS_COLLECTION]
        self.reviews_collection = MongoClient(Settings.MONGO_CONNECTION_STRING)[Settings.REVIEWS_DATABASE][Settings.REVIEWS_COLLECTION]
        self.reviewsCollection = MongoClient(Settings.MONGO_CONNECTION_STRING)[Settings.RAW_DATABASE][Settings.YELP_RAW_REVIEWS_COLLECTION]
        self.tags_collection = MongoClient(Settings.MONGO_CONNECTION_STRING)[Settings.TAGS_DATABASE][Settings.REVIEWS_COLLECTION]
        self.corpus_collection = MongoClient(Settings.MONGO_CONNECTION_STRING)[Settings.TAGS_DATABASE][Settings.CORPUS_COLLECTION]

        self.reviews_collection.remove(multi= True)
        MongoClient(Settings.MONGO_CONNECTION_STRING).drop_database(Settings.TAGS_DATABASE)
        self.IDS = IDS

    def clean_data(self):
        #resName = input('Restaurant Name: ')
        #city = input('City: ')

        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
            print('\nSuccessfully loaded pickle')

        model = load_model('./sentModel.h5')



        business_ids = self.IDS
        reviews_for_all_bussiness = {}

        '''
        for id in self.businessCollection.find({'name':resName, 'city':city},{'_id':0, 'business_id':1}):
            business_ids.append(id)

        print("IDs found = ", len(business_ids))
        
        '''
        print(business_ids)

        for i in business_ids:
            reviews_for_all_bussiness[i['business_id']] = [[x['review_id'], x['text'], x['stars']] for x in list(self.reviewsCollection.find({'business_id':i['business_id']},{'_id':0}))]

        #print(reviews_for_all_bussiness)
        all_negative_reviews=""

        for i in business_ids:
            texts = [x[1] for x in reviews_for_all_bussiness[i['business_id']]]
            #print('Reviews', texts)
            sequences = tokenizer.texts_to_sequences(texts)
            text_data = pad_sequences(sequences, maxlen=300)
            predictions = model.predict(text_data)
            #print(predictions)
            t=0
            for j in range(len(texts)):
                if predictions[j] <= 0.5:
                    self.reviews_collection.insert({
                        "reviewId": reviews_for_all_bussiness[i['business_id']][j][0],
                        "business": i['business_id'],
                        "text": reviews_for_all_bussiness[i['business_id']][j][1],
                        "stars": reviews_for_all_bussiness[i['business_id']][j][2]
                    })
                    all_negative_reviews += texts[j]
                    t+=1

            print('Negative Reviews found : ', t)
            if (t < 15):
                print("Very Few Negative Reviews. Skipping this Business/Branch ")
                #all_negative_reviews = ""
                return None

        print("Running reviews.py")
        #========================================
        #Reviews.py
        #========================================

        reviews_cursor = self.reviews_collection.find()
        reviewsCount = reviews_cursor.count()
        reviews_cursor.batch_size(1000)

        stopwords = {}
        with open('stopwords.txt', 'rU') as f:
            for line in f:
                stopwords[line.strip()] = 1

        done = 0
        start = time.time()

        for review in reviews_cursor:
            words = []
            sentences = nltk.sent_tokenize(review["text"].lower())

            for sentence in sentences:
                tokens = nltk.word_tokenize(sentence)
                text = [word for word in tokens if word not in stopwords]
                tagged_text = nltk.pos_tag(text)

                for word, tag in tagged_text:
                    words.append({"word": word, "pos": tag})

            self.tags_collection.insert({
                "reviewId": review["reviewId"],
                "business": review["business"],
                "text": review["text"],
                "words": words
            })


        print("Running Corpus.py")
        #==================================================
        #Corpus.py
        #==================================================

        reviews_cursor = self.tags_collection.find()
        reviewsCount = reviews_cursor.count()
        reviews_cursor.batch_size(5000)

        lem = nltk.stem.wordnet.WordNetLemmatizer()

        done = 0
        start = time.time()

        for review in reviews_cursor:
            nouns = []
            words = [word for word in review["words"] if word["pos"] in ["NN", "NNS"]]

            for word in words:
                nouns.append(lem.lemmatize(word["word"]))

            self.corpus_collection.insert({
                "reviewId": review["reviewId"],
                "business": review["business"],
                "text": review["text"],
                "words": nouns
            })

        return all_negative_reviews

class Corpus(object):
    def __init__(self, cursor, reviews_dictionary, corpus_path):
        self.cursor = cursor
        self.reviews_dictionary = reviews_dictionary
        self.corpus_path = corpus_path

    def __iter__(self):
        self.cursor.rewind()
        for review in self.cursor:
            yield self.reviews_dictionary.doc2bow(review["words"])

    def serialize(self):
        gensim.corpora.BleiCorpus.serialize(self.corpus_path, self, id2word=self.reviews_dictionary)

        return self


class Dictionary(object):
    def __init__(self, cursor, dictionary_path):
        self.cursor = cursor
        self.dictionary_path = dictionary_path

    def build(self):
        self.cursor.rewind()
        dictionary = gensim.corpora.Dictionary(review["words"] for review in self.cursor)
        dictionary.filter_extremes(keep_n=10000)
        dictionary.compactify()
        gensim.corpora.Dictionary.save(dictionary, self.dictionary_path)

        return dictionary


class Train:
    def __init__(self):
        pass

    @staticmethod
    def run(lda_model_path, corpus_path, num_topics, id2word):
        corpus = gensim.corpora.BleiCorpus(corpus_path)
        lda = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=id2word)
        lda.save(lda_model_path)

        return lda


class keyword_sentiment:
    def __init__(self,Neg_Review,Neg_keyword):
        self.Review = Neg_Review
        self.keyword = Neg_keyword

    def get_keyword_sentiment(self):

        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
            print('\nSuccessfully loaded pickle')

        model = load_model('./sentModel.h5')
        final_neg_keywords={}
        final_neg_keywords_comments={}

        for value in self.keyword:
            text = re.findall(r"([^.]*?" + value[0] +"[^.]*\.)", self.Review)

            sequences = tokenizer.texts_to_sequences(text)
            text_data = pad_sequences(sequences, maxlen=300)

            prediction = np.mean(model.predict(text_data))

            if prediction < 0.5:
                if value[0] in final_neg_keywords:
                    print("===========DUPLICATE DETECTED====")
                final_neg_keywords[value[0]] = prediction
                final_neg_keywords_comments[value[0]] = text


        return  sorted(final_neg_keywords.items(), key=operator.itemgetter(1)),final_neg_keywords_comments




def main():


    #GET USER INPUTS and extract  Business IDS for a specific restaurant or a Chain of Restaurants:
    resName = input('Restaurant Name: ')
    city = input('City: ')
    businessCollection = MongoClient(Settings.MONGO_CONNECTION_STRING)[Settings.RAW_DATABASE][Settings.YELP_RAW_BUSINESS_COLLECTION]
    business_data=[]
    for id in businessCollection.find({'name': resName, 'city': city}, {'_id': 0, 'business_id': 1, 'address':1}):
        business_data.append(id)

    print("IDs found = ", len(business_data))

    for bid in business_data:
        b_temp =[]
        b_temp.append(bid)
        Negative_Reviews = PreProcessing(b_temp).clean_data()

        if not Negative_Reviews:
            continue

        #MODEL GENERATION
        #logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        dictionary_path = "models/dictionary.dict"
        corpus_path = "models/corpus.lda-c"
        lda_num_topics = 10
        lda_model_path = "models/lda_model_50_topics.lda"

        corpus_collection = MongoClient(Settings.MONGO_CONNECTION_STRING)[Settings.TAGS_DATABASE][Settings.CORPUS_COLLECTION]
        reviews_cursor = corpus_collection.find()

        dictionary = Dictionary(reviews_cursor, dictionary_path).build()
        Corpus(reviews_cursor, dictionary, corpus_path).serialize()
        Train.run(lda_model_path, corpus_path, lda_num_topics, dictionary)


        #print('====Display the final output======')
        #logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        lda = gensim.models.LdaModel.load(lda_model_path)


        #EXTRACT THE KEYWORDS FROM THE TOPICS
        key_words = {}

        for i, topic in enumerate(lda.show_topics(num_topics=lda_num_topics)):
            #print('#%i: %s' % (i, str(topic)))
            key_list = topic[1].split('+')

            for vals in key_list:
                temp = vals.split('*')
                temp[1]= temp[1].strip(' ')
                temp[1] = temp[1].strip('\"')
                if temp[1] in key_words:
                    key_words[temp[1]] += float(temp[0])
                else:
                    key_words[temp[1]] = float(temp[0])

        neg_keyword = sorted(key_words.items(), key=operator.itemgetter(1), reverse=True)



        print("==============================================================================================")
        print("Business : ", resName, ", @Location : ", bid['address'])
        print("==============================================================================================")
        #Pass only first 10
        print("Possible Negative Keywords = \n", neg_keyword[0:10])

        #RERUN THE SENTIMENT ANALYSIS FOR THE KEYWORD SENTNECES IN THE NEGATIVE COMMENTS
        to_improve, neg_comments = keyword_sentiment(Negative_Reviews,neg_keyword[0:10]).get_keyword_sentiment()

        print("\nFinal Things to improve : \n", to_improve)
        print("\nSome COMMENTS REGARDING THEM")
        for z in range(len(to_improve)):
            print("for ", to_improve[z], " COMMENTS : \n", neg_comments[to_improve[z][0]][0:5], "\n")

        print("***********************************************************************************************")


if __name__ == '__main__':
    main()

