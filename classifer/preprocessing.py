import numpy as np
import torch
import torch.utils.data as Data
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import mmsdk
from mmsdk import mmdatasdk
from mmsdk.mmdatasdk import cmu_mosi
import argparse
import os
import sys

sys.path.append('../../tools/CMU-MultimodalSDK/')

H5_FILE = '../../tools/datasets/cmu-mosi/seq_length_20/X_train.h5'
DATA_PATH = './deploy'  # 如果要重新对齐，改为'./cmumosi'
ALIGN = False
DEPLOY = False


def myavg(intervals, features):
    return np.average(features, axis=0)


visual_field = 'CMU_MOSI_VisualFacet_4.1'
acoustic_field = 'CMU_MOSI_COVAREP'
text_field = 'CMU_MOSI_ModifiedTimestampedWords'
word_field = 'CMU_MOSI_TimestampedWordVectors_1.1'
label_field = 'CMU_MOSI_Opinion_Labels'

features = [
    # text_field,
    word_field,
    visual_field,
    acoustic_field,
]
recipe = {feat: os.path.join(DATA_PATH, feat) + '.csd' for feat in features}


def load_data():
    cmumosi_highlevel = mmdatasdk.mmdataset(recipe)
    if ALIGN:
        cmumosi_highlevel.align(word_field, collapse_functions=[myavg])

    cmumosi_highlevel.add_computational_sequences({label_field: os.path.join(DATA_PATH, label_field) + '.csd'},
                                                  destination=None)
    if ALIGN:
        cmumosi_highlevel.align(label_field)
        if DEPLOY:
            deploy_files = {x: x for x in cmumosi_highlevel.computational_sequences.keys()}
            cmumosi_highlevel.deploy('./deploy', deploy_files)
    return cmumosi_highlevel


def multi_collate(batch):
    """对数据进行整理，用于后续神经网络处理"""
    # 重排序用于rnn
    batch = sorted(batch, key=lambda x: x[0][0].shape[0], reverse=True)
    labels = torch.cat([torch.from_numpy(sample[1])for sample in batch])
    word = pad_sequence([torch.FloatTensor(sample[0][0]) for sample in batch])
    visual = pad_sequence([torch.FloatTensor(sample[0][1]) for sample in batch])
    acoustic = pad_sequence([torch.FloatTensor(sample[0][2]) for sample in batch])
    lengths = torch.LongTensor([len(sample[0][0]) for sample in batch])

    return word, visual, acoustic, labels, lengths

def multi_collate2(batch):
    """对数据进行整理，用于后续神经网络处理"""
    # 重排序用于rnn
    batch = sorted(batch, key=lambda x: x[2], reverse=True)
    labels = torch.cat([torch.from_numpy(sample[1][:sample[2]])for sample in batch])
    lengths = torch.cat([torch.from_numpy(sample[2].reshape(1)) for sample in batch])
    word = pad_sequence([torch.FloatTensor(sample[0][0]) for sample in batch])
    visual = pad_sequence([torch.FloatTensor(sample[0][1]) for sample in batch])
    acoustic = pad_sequence([torch.FloatTensor(sample[0][2]) for sample in batch])

    return word, visual, acoustic, labels, lengths

def split_data(dataset):
    """划分并处理数据"""
    train = []
    valid = []
    test = []
    train_fold = cmu_mosi.cmu_mosi_std_folds.standard_train_fold
    valid_fold = cmu_mosi.cmu_mosi_std_folds.standard_valid_fold
    test_fold = cmu_mosi.cmu_mosi_std_folds.standard_test_fold

    for segment in dataset[label_field].keys():
        vid = segment.strip().split('[')[0]     # video ID
        label = dataset[label_field][segment]['features']
        word = dataset[word_field][segment]['features']
        visual = dataset[visual_field][segment]['features']
        acoustic = dataset[acoustic_field][segment]['features']

        # print(word.shape)
        # print(acoustic.shape)
        # print(visual.shape)
        # print(label.shape)
        if not word.shape[0] == visual.shape[0] == acoustic.shape[0]:
            cut_len = min(word.shape[0], visual.shape[0], acoustic.shape[0])
            word = word[:cut_len, :]
            visual = visual[:cut_len, :]
            acoustic = acoustic[:cut_len, :]

        # remove nan values
        label = np.nan_to_num(label)
        word = np.nan_to_num(word)
        visual = np.nan_to_num(visual)
        acoustic = np.nan_to_num(acoustic)

        # z-normalization per instance and remove nan/infs
        # word = np.nan_to_num((word - word.mean(0, keepdims=True))/np.std(word, axis=0, keepdims=True))
        visual = np.nan_to_num((visual - visual.mean(0, keepdims=True))/np.std(visual, axis=0, keepdims=True))
        acoustic = np.nan_to_num((acoustic - acoustic.mean(0, keepdims=True))/np.std(acoustic, axis=0, keepdims=True))

        if vid in train_fold:
            train.append(((word, visual, acoustic), label, segment))
        elif vid in valid_fold:
            valid.append(((word, visual, acoustic), label, segment))
        elif vid in test_fold:
            test.append(((word, visual, acoustic), label, segment))
        else:
            print("segment error")

    # mosi_data = {'train': train, 'valid': valid, 'test': test}
    # np.save('mosi.npy', mosi_data)
    print("train size: %s" % len(train))
    print("valid size: %s" % len(valid))
    print("test size: %s" % len(test))

    train_loader = Data.DataLoader(train, batch_size=24, shuffle=True, collate_fn=multi_collate)
    valid_loader = Data.DataLoader(valid, batch_size=24, shuffle=True, collate_fn=multi_collate)
    test_loader = Data.DataLoader(test, batch_size=24, shuffle=True, collate_fn=multi_collate)

    return train_loader, valid_loader, test_loader


def load_pickle_data():
    import pickle
    multi_data = {'text':[], 'video':[], 'audio':[]}
    train = []
    valid = []
    test = []
    for mode in ['text', 'video', 'audio']:
        with open('../data/mosi/utterance/' + mode + '.pickle', 'rb') as handle:
            (train_data, train_label, test_data, test_label, maxlen, train_length, test_length) = pickle.load(
                handle, encoding='latin1')
            train_data, valid_data, train_label, valid_label, train_length, valid_length = train_test_split(
                train_data, train_label, train_length, test_size=10)
            all_data = [train_data, valid_data, train_label, valid_label, train_length, valid_length, test_data,
                        test_label, test_length]
            for i in range(len(all_data)):
                all_data[i] = [sample.squeeze() for sample in np.split(np.array(all_data[i]), len(all_data[i]))]
            train_data, valid_data, train_label, valid_label, train_length, valid_length, test_data, test_label,\
            test_length = all_data
            multi_data[mode].append((train_data, train_label, train_length))
            multi_data[mode].append((valid_data, valid_label, valid_length))
            multi_data[mode].append((test_data, test_label, test_length))
    for i in range(len(multi_data['text'][0][0])):
        train.append([(multi_data['text'][0][0][i], multi_data['video'][0][0][i], multi_data['audio'][0][0][i]),
                      multi_data['text'][0][1][i], multi_data['text'][0][2][i]])
    for j in range(len(multi_data['text'][1][0])):
        valid.append([(multi_data['text'][1][0][j], multi_data['video'][1][0][j], multi_data['audio'][1][0][j]),
                      multi_data['text'][1][1][j], multi_data['text'][1][2][j]])
    for k in range(len(multi_data['text'][2][0])):
        test.append([(multi_data['text'][2][0][k], multi_data['video'][2][0][k], multi_data['audio'][2][0][k]),
                      multi_data['text'][2][1][k], multi_data['audio'][2][2][k]])

    train_loader = Data.DataLoader(train, batch_size=32, shuffle=True, collate_fn=multi_collate2, )
    valid_loader = Data.DataLoader(valid, batch_size=32, shuffle=True, collate_fn=multi_collate2)
    test_loader = Data.DataLoader(test, batch_size=32, shuffle=True, collate_fn=multi_collate2)

    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    # load_data()
    load_pickle_data()