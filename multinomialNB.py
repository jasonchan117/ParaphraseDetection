from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score, recall_score, f1_score,confusion_matrix,accuracy_score
from utils import *
import numpy as np
from nltk.metrics.distance import edit_distance

train_data, train_label=getData('data/train.data')
vali_data, vali_label=getData('data/dev.data')

for i in range(len(train_data)):
    train_data[i]=''.join([train_data[i][0],' ',train_data[i][1]])


for i in range(len(vali_data)):
    vali_data[i]=''.join([vali_data[i][0],' ',vali_data[i][1]])


cv = CountVectorizer()
x = cv.fit_transform(train_data)
m = MultinomialNB(fit_prior=True)
m.fit(x, train_label)
np.set_printoptions(threshold=np.inf)
vali_res=m.predict(cv.transform(vali_data))
print(precision_score(vali_label, vali_res), recall_score(vali_label, vali_res), f1_score(vali_label, vali_res),accuracy_score(vali_label, vali_res))