# used bag of words on the text and then used supervised learning on the integer values I got from it
import json
import re
import nltk 
import heapq
from nltk.corpus import stopwords
import numpy as np
import sklearn
from sklearn import svm



rating = []
comments = []
overall = []

for line in open('Musical_Instruments_ratings.json', 'r'):
    rating.append(json.loads(line))
    
for i in range(len (rating)):
    comments+= [rating[i]["reviewText"]] 
    overall+= [rating[i]["overall"]] #overall rating (1:5)

for i in range(len(comments )):
    comments [i] = comments[i].lower()
    comments [i] = re.sub(r'\W',' ',comments [i]) # for deleting quotation marks
    comments [i] = re.sub(r'\s+',' ',comments [i])# for deleting quotation marks
    
stop_words = set(stopwords.words('english')) # common words like the, I, a, etc...

good_rating = {}
'''bad_rating = {}'''

for i in range(len(comments)):
    comment = comments[i]
    tokens = nltk.word_tokenize(comment)
    if (overall[i] >= 3):
        for token in tokens:
            if token in  stop_words:
                continue
            if token not in good_rating.keys():
                good_rating[token] = 1
            else:
                good_rating[token] += 1
    """else:
        for token in tokens:
            if token in  stop_words:
                continue
            if token not in bad_rating.keys():
                bad_rating[token] = 1
            else:
                bad_rating[token] += 1"""
                
most_freq_good = heapq.nlargest(500, good_rating, key=good_rating.get) #most frequent words in the dictionary
most_freq_bad = heapq.nlargest(20, bad_rating, key=bad_rating.get)
sentence_vectors = []
for sentence in comments: # resaerch sentence vectors in bag of words
    sentence_tokens = nltk.word_tokenize(sentence)
    sent_vec = []
    for token in most_freq_good:
        if token in sentence_tokens:
            sent_vec.append(1)
        else:
            sent_vec.append(0)
    sentence_vectors.append(sent_vec)
common = []
for i in range(len(sentence_vectors)):
    sent = sentence_vectors[i] 
    ones = 0
    for j in sent:
        if (j==1): 
            ones+= 1
    if (len(comment)!=0):
        common += [(ones/len(comment))*100]
    else:
        common += [0]
B = np.reshape(common, (len(overall), 1))
X = np.array(B)
y = np.array(overall)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)
model = svm.SVC(kernel='linear')

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
predictions = model.predict(x_test)

print ("Accuracy is equal to", acc)
for i in range(len(x_test)):
    print (x_test[i], predictions[i], y_test[i])
    

