from torchtext import data
from utils import *
from model import RNN
import torch
import torch.nn as nn
import time
import torch.optim as optim
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--traindata', default='data/train.data',
                    help='The path to training data.')
parser.add_argument('--validata',default='data/dev.data',
                    help='The path to validation data.')
# Training settings
parser.add_argument('--epochs', default=10, type=int,
                    help='epochs to train')
parser.add_argument('--dropout', default=0.5,
                    help='Dropout rate of RNN.')
parser.add_argument('--lr', default=1e-4, type=float,
                    help='learning rate')
parser.add_argument('--batch', default=64)
parser.add_argument('--weight_decay', default=0, type=float,
                    help='factor for L2 regularization')
parser.add_argument('--seed', default=594277, type=int,
                    help='manual seed')
parser.add_argument('--cuda', action='store_true',
                    help='Use cuda or not')
# RNN settings
parser.add_argument('--embed_dim', default=100,
                    help='The dimension of embedding.')
parser.add_argument('--hidden_dim', default=256,
                    help='The dimension of hidden layer.')
parser.add_argument('--layer',default=2,
                    help='The number of RNN layers.')
parser.add_argument('--bid',action='store_true',
                    help='RNN is bidirectional or not.')
args = parser.parse_args()


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

SEED = args.seed
torch.manual_seed(SEED)
if args.cuda == True:
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
# Load word embedding
TEXT.build_vocab(train, max_size=25000, vectors="glove.6B.100d", unk_init=torch.Tensor.normal_)
LABEL.build_vocab(train)

BATCH_SIZE = args.batch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_iterator, valid_iterator = data.BucketIterator.splits(
    (train, valid),
    batch_size = BATCH_SIZE,
    sort_key= lambda x : len(x.text),
    sort_within_batch=False,
    device = device
)
INPUT_DIM = len(TEXT.vocab)  # 25002
EMBEDDING_DIM = args.embed_dim
HIDDEN_DIM = args.hidden_dim
OUTPUT_DIM = 1
N_LAYERS = args.layer
BIDIRECTIONAL = args.bid
DROPOUT = args.dropout
# The index of token 'pad'
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
# Initialize the model
model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX)

pretrained_embeddings = TEXT.vocab.vectors
# Initialize the model with word embedding
model.embedding.weight.data.copy_(pretrained_embeddings)

UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]  # UNK_IDX = 0

# Initial the vectors 'unknown' and 'pad' as 0.
model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)


# Training model
optimizer = optim.Adam(model.parameters(),lr=args.lr, weight_decay=args.weight_decay)
criterion = nn.BCEWithLogitsLoss()
model = model.to(device)
criterion = criterion.to(device)
N_EPOCHS = args.epochs
best_valid_loss = float('inf')  # 初试的验证集loss设置为无穷大
for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss, train_acc = training(model, train_iterator, optimizer, criterion)
    # Derives the metric values including accuracy, precision, recall and f1.
    valid_loss, valid_acc, valid_prec, valid_recall, valid_f1 = evaluate(model, valid_iterator, criterion)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    # Save the model.
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'wordavg-model.pt')

    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc * 100:.2f}% | Val. Prec: {valid_prec * 100:.2f}% | Val. Recall: {valid_recall * 100:.2f}% | Val. F1: {valid_f1 * 100:.2f}%')