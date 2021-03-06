from torchtext import data
from utils import *
from model import RNN
import torch
import torch.nn as nn
import time

tokenize = lambda x: x.split()
TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True, fix_length=200)
LABEL = data.Field(sequential=False, use_vocab=False,dtype=torch.float)

train_data,train_label=getData('data/train.data')
vali_data,vali_label=getData('data/dev.data')
for i in range(len(train_data)):
    train_data[i]=''.join([train_data[i][0],' ',train_data[i][1]])
    if train_label[i]==-1:
        train_label[i]+=1
for i in range(len(vali_data)):
    vali_data[i]=''.join([vali_data[i][0],' ',vali_data[i][1]])
    if vali_label[i]==-1:
        vali_label[i]+=1

train_examples, train_fields = get_dataset(train_data,train_label, TEXT, LABEL, data=data)
valid_examples, valid_fields = get_dataset(vali_data,vali_label, TEXT, LABEL, data=data)
# Build training and validation datasets
train = data.Dataset(train_examples, train_fields)
valid = data.Dataset(valid_examples, valid_fields)

SEED = 1234
torch.manual_seed(SEED)  # 为CPU设置随机种子
# Load word embedding
TEXT.build_vocab(train, max_size=25000, vectors="glove.6B.100d", unk_init=torch.Tensor.normal_)
LABEL.build_vocab(train)



BATCH_SIZE = 16

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 相当于把样本划分batch，知识多做了一步，把相等长度的单词尽可能的划分到一个batch，不够长的就用padding。
train_iterator, valid_iterator = data.BucketIterator.splits(
    (train, valid ),
    batch_size = BATCH_SIZE,
    device = device
)



INPUT_DIM = len(TEXT.vocab)  # 25002
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
# PAD_IDX = 1 为pad的索引
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
# Initialize the model
model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX)

pretrained_embeddings = TEXT.vocab.vectors
# Initialize with word embedding
model.embedding.weight.data.copy_(pretrained_embeddings)

UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]  # UNK_IDX = 0

# 词汇表25002个单词，前两个unk和pad也需要初始化，把它们初始化为0
model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

import torch.optim as optim
# Train model
# 定义优化器
optimizer = optim.Adam(model.parameters())

# 定义损失函数，这个BCEWithLogitsLoss特殊情况，二分类损失函数
criterion = nn.BCEWithLogitsLoss()

# 送到GPU上去
model = model.to(device)
criterion = criterion.to(device)

N_EPOCHS = 10

best_valid_loss = float('inf')  # 初试的验证集loss设置为无穷大

for epoch in range(N_EPOCHS):
    start_time = time.time()

    train_loss, train_acc = training(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc, valid_prec, valid_recall, valid_f1 = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    # 只要模型效果变好，就存模型(参数)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'wordavg-model.pt')

    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc * 100:.2f} | Val. Prec: {valid_prec * 100:.2f} | {valid_recall * 100:.2f} | {valid_f1 * 100:.2f}%')