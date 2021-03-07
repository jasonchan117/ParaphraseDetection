from tqdm import tqdm
import torch
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score
import torch.nn as nn
import torch.nn.functional as F

def loss_function(gt,pre):
  flag=1
  for g,p in zip(gt,pre):
    if g.cpu().item == 0.:
        loss =F.binary_cross_entropy_with_logits(p,g) * (3996/7534)
    else:
        loss = F.binary_cross_entropy_with_logits(p,g)
    if flag==1:
      sum_loss=loss
      flag=0
    else:
      sum_loss+=loss
  return sum_loss
def getData(path):
    data=[]
    label=[]
    for line in open(path):
        if float(line.split('\t')[4][4]) - float(line.split('\t')[4][1]) == 1.0 :
            continue
        line = line.strip()
        #read in training or dev data with labels
        if len(line.split('\t')) == 7:
            data.append([line.split('\t')[2],line.split('\t')[3]])
            label.append(1 if float(line.split('\t')[4][1]) > float(line.split('\t')[4][4]) else -1)
        #read in test data without labels
        elif len(line.split('\t')) == 6:
            data.append([line.split('\t')[2],line.split('\t')[3]])
        else:
            continue

    return data, label

# Construct example and field objects.
def get_dataset(data_x,label, text_field, label_field, data,test=False):

    fields = [("id", None),("text", text_field), ("label", label_field)]
    examples = []

    if test:
        for text in tqdm(data_x):
            examples.append(data.Example.fromlist([None, text, None], fields))
    else:
        for text, label in tqdm(zip(data_x, label)):
            examples.append(data.Example.fromlist([None, text, label], fields))
    return examples, fields


def binary_accuracy(preds, y):

    rounded_preds = torch.round(torch.sigmoid(preds))

    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)

    return acc


def training(model, iterator, optimizer):
    epoch_loss = 0
    epoch_acc = 0
    total_len = 0

    model.train()

    for batch in iterator:

        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1)

        loss = loss_function( batch.label,predictions)

        acc = binary_accuracy(predictions, batch.label)

        loss.backward()  
        optimizer.step() 
        epoch_loss += loss.item() * len(batch.label)
        epoch_acc += acc.item() * len(batch.label)

        total_len += len(batch.label)

    return epoch_loss / total_len, epoch_acc / total_len

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    total_len = 0

    model.eval()

    with torch.no_grad():

        ind=0
        sum_prec=0.
        sum_recall=0.
        sum_f1=0.
        sum_acc=0.
        for batch in iterator:

            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = accuracy_score(batch.label.cpu(), torch.round(torch.sigmoid(predictions)).cpu())
            prec = precision_score(batch.label.cpu(), torch.round(torch.sigmoid(predictions)).cpu())
            recall = recall_score(batch.label.cpu(), torch.round(torch.sigmoid(predictions)).cpu())
            f1= f1_score(batch.label.cpu(), torch.round(torch.sigmoid(predictions)).cpu())
            sum_prec+=prec
            sum_f1+=f1
            sum_recall+=recall
            sum_acc+=acc

            epoch_loss += loss.item() * len(batch.label)
            total_len += len(batch.label)

            ind+=1
    model.train()

    return epoch_loss / total_len, sum_acc/ind , sum_prec / ind, sum_recall / ind , sum_f1 / ind

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

