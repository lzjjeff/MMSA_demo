import sys
sys.path.append('../')
import numpy as np
import time
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from classifer.preprocessing import load_data, split_data
from classifer.config import *
# from model.mfn import MFN


torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
CUDA = torch.cuda.is_available()


def metrics(y_true, y_pred):
    y_true_bin = y_true >= 0
    y_pred_bin = y_pred >= 0

    bi_acc = accuracy_score(y_true_bin, y_pred_bin)
    f1 = f1_score(y_true_bin, y_pred_bin)
    multi_acc = np.round(sum(np.round(y_pred) == np.round(y_true)) / float(len(y_true)), 7)[0]
    return bi_acc, f1, multi_acc


# define some model settings and hyper-parameters

# load and split data
multi_dataset = load_data()
train_loader, valid_loader, test_loader = split_data(multi_dataset)

# initial our model
if model_name is 'marn':
    model = MODEL[model_name](input_dims, hidden_dims, output_dim, local_dims, att_num, dropout, modal=MODAL)
elif model_name is 'lmf':
    model = MODEL[model_name](input_dims, hidden_dims, output_dim, fc_dim, rank, dropout, modal=MODAL)
elif model_name is 'mmmu_ba':
    model = MODEL[model_name](input_dims, hidden_dims, output_dim, fc_dim, rnn_dropout, dropout, modal=MODAL)
else:
    model = MODEL[model_name](input_dims, hidden_dims, output_dim, fc_dim, dropout, modal=MODAL)
if CUDA:
    model.cuda()
optimizer = optim.Adam([param for param in model.parameters() if param.requires_grad], lr=lr, weight_decay=weight_decay)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
lr_scheduler.step()
loss_func = nn.L1Loss(reduction='sum')

tic = time.time()
# start training and validation
if TRAIN:
    with open(f'{RESULT_PATH}result_%s_%s' % ('_'.join([_modal[0] for _modal in MODAL]), model_name), 'a', encoding='utf-8') as fo:
        fo.write("=" * 60)
    train_losses = []
    valid_losses = []
    best_valid_loss = float('inf')
    for epoch in range(EPOCH):
        lrs = []
        model.train()
        train_loss = 0.0
        train_size = 0
        for batch in train_loader:
            model.zero_grad()
            w, v, a, y, l = batch
            batch_size = w.size(1)  # （length, batch_size, dim）
            train_size += batch_size
            if CUDA:
                w = w.cuda()
                v = v.cuda()
                a = a.cuda()
                y = y.cuda()
                l = l.cuda()
            y_pred = model({'word': w, 'visual': v, 'acoustic': a}, l)
            loss = loss_func(y_pred, y)
            loss.backward()
            # nn.utils.clip_grad_value_([param for param in model.parameters() if param.requires_grad], grad_clip_value)  # 梯度裁剪
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss / train_size
        train_losses.append(train_loss)
        print("EPOCH %s | Train loss: %s" % (epoch, round(train_loss, 4)))

        model.eval()
        with torch.no_grad():    # 不进行梯度计算
            valid_loss = 0.0
            valid_size = 0
            for batch in valid_loader:
                model.zero_grad()
                w, v, a, y, l = batch
                valid_size += w.size(1)
                if CUDA:
                    w = w.cuda()
                    v = v.cuda()
                    a = a.cuda()
                    y = y.cuda()
                    l = l.cuda()
                y_pred = model({'word': w, 'visual': v, 'acoustic': a}, l)
                loss = loss_func(y_pred, y)
                valid_loss += loss.item()
        valid_loss = valid_loss / valid_size
        valid_losses.append(valid_loss)
        print("EPOCH %s | Validtation loss: %s" % (epoch, round(valid_loss, 4)))
        if valid_loss <= best_valid_loss:
            best_valid_loss = valid_loss
            print("A new best model on valid set")
            torch.save(model.state_dict(), f'{SAVE_PATH}model_%s.std' % '_'.join([_modal[0] for _modal in MODAL]))
            torch.save(optimizer.state_dict(), f'{SAVE_PATH}optim_%s.std' % '_'.join([_modal[0] for _modal in MODAL]))
            lr_scheduler.step()
            print("Current learning rate: %s" % optimizer.state_dict()['param_groups'][0]['lr'])
            with open(f'{RESULT_PATH}result_%s_%s' % ('_'.join([_modal[0] for _modal in MODAL]), model_name), 'a', encoding='utf-8') as fo:
                fo.write("\nTrain loss: %s\nValidtation loss: %s" % (round(train_loss, 4), round(valid_loss, 4)))

        lrs.append(optimizer.state_dict()['param_groups'][0]['lr'])

# testing
if TEST:
    model.load_state_dict(torch.load(f'{SAVE_PATH}model_%s.std' % '_'.join([_modal[0] for _modal in MODAL])))
    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        test_size = 0
        for batch in test_loader:
            model.zero_grad()
            w, v, a, y, l = batch
            test_size += w.size(1)
            if CUDA:
                w = w.cuda()
                v = v.cuda()
                a = a.cuda()
                y = y.cuda()
                l = l.cuda()
            _y_pred = model({'word': w, 'visual': v, 'acoustic': a}, l)
            loss = loss_func(_y_pred, y)
            y_true.append(y.cpu())
            y_pred.append(_y_pred.cpu())
            test_loss += loss.item()
    print("Test loss: %s" % round(test_loss/test_size, 4))
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    bi_acc, f1, multi_acc = metrics(y_true, y_pred)
    print("Test set acc_2 is %s\nTest set f1 score is %s\nTest set acc_7 is %s" % (bi_acc, f1, multi_acc))
    with open(f'{RESULT_PATH}result_%s_%s' % ('_'.join([_modal[0] for _modal in MODAL]), model_name), 'a', encoding='utf-8') as fo:
        fo.write("\nTest loss: %s\nTest set accuracy is %s\nTest set f1 score is %s\nTest set acc_7 is %s\n"
                 % (round(test_loss/test_size, 4), bi_acc, f1, multi_acc))
tic = time.time()-tic
print(round(tic, 4), 's')

