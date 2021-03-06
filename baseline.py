from nltk.metrics.distance import edit_distance
from sklearn.metrics import precision_score, recall_score, f1_score,confusion_matrix,accuracy_score
from utils import *

train_data, train_label=getData('data/train.data')
vali_data, vali_label=getData('data/dev.data')

res = []
ed=[]


for i in vali_data:
    if edit_distance(i[0],i[1]) >=40:
        res.append(1)
    else:
        res.append(-1)
print(precision_score(vali_label, res), recall_score(vali_label, res), f1_score(vali_label, res),accuracy_score(vali_label, res))