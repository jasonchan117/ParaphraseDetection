from tqdm import tqdm
import torch
from sklearn.metrics import precision_score, recall_score, f1_score,confusion_matrix,accuracy_score
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
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    # .round函数 四舍五入，rounded_preds要么为0，要么为1
    # neg为0, pos为1
    rounded_preds = torch.round(torch.sigmoid(preds))

    # convert into float for division
    """
    a = torch.tensor([1, 1])
    b = torch.tensor([1, 1])
    print(a == b)
    output: tensor([1, 1], dtype=torch.uint8)
  
    a = torch.tensor([1, 0])
    b = torch.tensor([1, 1])
    print(a == b)
    output: tensor([1, 0], dtype=torch.uint8)
    """
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)

    return acc


def training(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    total_len = 0

    # model.train()代表了训练模式
    # model.train() ：启用 BatchNormalization 和 Dropout
    # model.eval() ：不启用 BatchNormalization 和 Dropout
    model.train()

    # iterator为train_iterator
    for batch in iterator:
        # 梯度清零，加这步防止梯度叠加
        optimizer.zero_grad()

        # batch.text 就是上面forward函数的参数text
        # 压缩维度，不然跟 batch.label 维度对不上

        predictions = model(batch.text).squeeze(1)

        loss = criterion(predictions, batch.label)

        acc = binary_accuracy(predictions, batch.label)

        loss.backward()  # 反向传播
        optimizer.step()  # 梯度下降

        # loss.item() 以及本身除以了 len(batch.label)
        # 所以得再乘一次，得到一个batch的损失，累加得到所有样本损失
        epoch_loss += loss.item() * len(batch.label)

        # (acc.item(): 一个batch的正确率) * batch数 = 正确数
        # train_iterator 所有batch的正确数累加
        epoch_acc += acc.item() * len(batch.label)

        # 计算 train_iterator 所有样本的数量，应该是17500
        total_len += len(batch.label)

    # epoch_loss / total_len ：train_iterator所有batch的损失
    # epoch_acc / total_len ：train_iterator所有batch的正确率
    return epoch_loss / total_len, epoch_acc / total_len


# 不用优化器了
def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    total_len = 0

    # 转成测试模式，冻结dropout层或其他层
    model.eval()

    with torch.no_grad():
        # iterator为valid_iterator
        ind=0
        sum_prec=0.
        sum_recall=0.
        sum_f1=0.
        for batch in iterator:
            # 没有反向传播和梯度下降

            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)
            prec = precision_score(batch.label, predictions)
            recall = recall_score(batch.label, predictions)
            f1= f1_score(batch.label, predictions)
            sum_prec+=prec
            sum_f1+=f1
            sum_recall+=recall

            epoch_loss += loss.item() * len(batch.label)
            epoch_acc += acc.item() * len(batch.label)
            total_len += len(batch.label)

            ind+=1
    # 调回训练模式
    model.train()

    return epoch_loss / total_len, epoch_acc / total_len, sum_prec/ind, sum_recall/ind,sum_f1/ind

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
