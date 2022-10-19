import numpy as np
import torch
import sys
import pandas as pd
from sklearn import preprocessing
# from keras_preprocssing.texte import Tokenizer
gene_map = {
    'A': [1, 0, 0, 0],
    'C': [0, 1, 0, 0],
    'G': [0, 0, 1, 0],
    'T': [0, 0, 0, 1],
    'N': [0, 0, 0, 0],

}
# samples=['NNN', 'NNA', 'NNC', 'NNG', 'NNT', 'NAN', 'NAA', 'NAC', 'NAG', 'NAT', 'NCN', 'NCA', 'NCC', 'NCG', 'NCT', 'NGN', 'NGA', 'NGC', 'NGG', 'NGT', 'NTN', 'NTA', 'NTC', 'NTG', 'NTT', 'ANN', 'ANA', 'ANC', 'ANG', 'ANT', 'AAN', 'AAA', 'AAC', 'AAG', 'AAT', 'ACN', 'ACA', 'ACC', 'ACG', 'ACT', 'AGN', 'AGA', 'AGC', 'AGG', 'AGT', 'ATN', 'ATA', 'ATC', 'ATG', 'ATT', 'CNN', 'CNA', 'CNC', 'CNG', 'CNT', 'CAN', 'CAA', 'CAC', 'CAG', 'CAT', 'CCN', 'CCA', 'CCC', 'CCG', 'CCT', 'CGN', 'CGA', 'CGC', 'CGG', 'CGT', 'CTN', 'CTA', 'CTC', 'CTG', 'CTT', 'GNN', 'GNA', 'GNC', 'GNG', 'GNT', 'GAN', 'GAA', 'GAC', 'GAG', 'GAT', 'GCN', 'GCA', 'GCC', 'GCG', 'GCT', 'GGN', 'GGA', 'GGC', 'GGG', 'GGT', 'GTN', 'GTA', 'GTC', 'GTG', 'GTT', 'TNN', 'TNA', 'TNC', 'TNG', 'TNT', 'TAN', 'TAA', 'TAC', 'TAG', 'TAT', 'TCN', 'TCA', 'TCC', 'TCG', 'TCT', 'TGN', 'TGA', 'TGC', 'TGG', 'TGT', 'TTN', 'TTA', 'TTC', 'TTG', 'TTT']
# tokenizer=Tokenizer(num_words=126)#创建一个分词器，设置只考虑前1000个单词
# tokenizer.fit_on_texts(samples)#构建单词索引

# def split_str(s, length, num):
#     str = []
#     for i in range(num):
#         str.append(s[i:i+length])
#     return str
#
# def Get_DNA_Sequence(TF_name):
#     X_train = []
#     y_train = []
#     z_train = []
#     zero_vector = [0., 0., 0., 0., 0., 0., 0., 0.]
#     pos_file = open("./sequence/"+TF_name+"_pos.fasta",'r')
#     neg_file = open("./sequence/"+TF_name+"_neg_1x.fasta",'r')
#     sample = []
#     pos_num = 0
#     neg_num = 0
#     length = 101
#     number = 0
#     # print("READ DNA for %s" % (cell))
#     for line in pos_file:
#
#         size = 0
#         line = line.strip('\n\r')
#
#         if len(line) < 5:
#             break
#         if line[0] == ">":
#             i = 0
#             continue
#
#         else:
#             line = line.upper()
#             if i == 0:
#                content = line
#                i = i + 1
#                continue
#             else:
#                content = content+line
#
#
#         # if number < 511:
#         #     number = number + 1
#         #     continue
#
#         if len(content) > length:
#             size = length
#         else:
#             size = len(content)
#         # row = np.random.randn(101, 4)
#         content = 'N' + content + 'N'
#         content1 = split_str(content,3 ,101)
#         one_hot_results = tokenizer.texts_to_matrix(content1, mode='binary')
#         one_hot_results = one_hot_results[:,0:125]
#         X_train.append(one_hot_results)
#         # for location, base in enumerate(range(0,size), start=0):
#         #     row[location] = gene_map[content[base]]
#         # X_train.append(row)
#         pos_num = pos_num + 1
#         y_train.append(1)
#
#     for line in neg_file:
#
#         size = 0
#         line = line.strip('\n\r')
#         if len(line) < 5:
#             break
#         if line[0] == ">":
#             i = 0
#             continue
#
#         else:
#             if i == 0:
#                content = line
#                i = i + 1
#                continue
#             else:
#                content = content + line
#
#         if len(content) > length:
#             size = length
#         else:
#             size = len(content)
#         content = 'N' + content + 'N'
#         content1 = split_str(content,3 ,101)
#         one_hot_results = tokenizer.texts_to_matrix(content1, mode='binary')
#         one_hot_results = one_hot_results[:,0:125]
#         X_train.append(one_hot_results)
#         # row = np.random.randn(101, 4)
#         # for location, base in enumerate(range(0,size), start=0):
#         #     row[location] = gene_map[content[base]]
#         # X_train.append(row)
#         neg_num = neg_num + 1
#         y_train.append(0)
#
#     print("the number of positive train sample: %d" % pos_num)
#     print("the number of negative train sample: %d" % neg_num)
#     X_train = np.array(X_train,dtype="float32")
#     y_train = np.array(y_train,dtype="float32")
#     np.random.seed(1)
#     shuffle_ix = np.random.permutation(np.arange(len(X_train)))
#     data = X_train[ shuffle_ix ]
#     label = y_train[ shuffle_ix ]
#     return data, label


f = pd.read_csv('./weight2.csv')
f = f.to_dict('list')

gene_map = {
    'A': [1, 0, 0, 0],
    'C': [0, 1, 0, 0],
    'G': [0, 0, 1, 0],
    'T': [0, 0, 0, 1],
    'N': [0, 0, 0, 0],

}
# samples=['NNN', 'NNA', 'NNC', 'NNG', 'NNT', 'NAN', 'NAA', 'NAC', 'NAG', 'NAT', 'NCN', 'NCA', 'NCC', 'NCG', 'NCT', 'NGN', 'NGA', 'NGC', 'NGG', 'NGT', 'NTN', 'NTA', 'NTC', 'NTG', 'NTT', 'ANN', 'ANA', 'ANC', 'ANG', 'ANT', 'AAN', 'AAA', 'AAC', 'AAG', 'AAT', 'ACN', 'ACA', 'ACC', 'ACG', 'ACT', 'AGN', 'AGA', 'AGC', 'AGG', 'AGT', 'ATN', 'ATA', 'ATC', 'ATG', 'ATT', 'CNN', 'CNA', 'CNC', 'CNG', 'CNT', 'CAN', 'CAA', 'CAC', 'CAG', 'CAT', 'CCN', 'CCA', 'CCC', 'CCG', 'CCT', 'CGN', 'CGA', 'CGC', 'CGG', 'CGT', 'CTN', 'CTA', 'CTC', 'CTG', 'CTT', 'GNN', 'GNA', 'GNC', 'GNG', 'GNT', 'GAN', 'GAA', 'GAC', 'GAG', 'GAT', 'GCN', 'GCA', 'GCC', 'GCG', 'GCT', 'GGN', 'GGA', 'GGC', 'GGG', 'GGT', 'GTN', 'GTA', 'GTC', 'GTG', 'GTT', 'TNN', 'TNA', 'TNC', 'TNG', 'TNT', 'TAN', 'TAA', 'TAC', 'TAG', 'TAT', 'TCN', 'TCA', 'TCC', 'TCG', 'TCT', 'TGN', 'TGA', 'TGC', 'TGG', 'TGT', 'TTN', 'TTA', 'TTC', 'TTG', 'TTT']
# tokenizer=Tokenizer(num_words=126)#创建一个分词器，设置只考虑前1000个单词
# tokenizer.fit_on_texts(samples)#构建单词索引

def split_str(s, length, num):
    str = []
    for i in range(num):
        str.append(s[i:i+length])
    return str
def Get_Conservation_Score(TF_name):
    a = pd.read_table('./data/sequence/'+TF_name+"_pos_cs.fasta", sep=' ', header=None)
    a.iloc[:,-1] = a.mean(1)
    b = pd.read_table('./data/sequence/' +TF_name+"_neg_1x_cs.fasta", sep=' ', header=None)
    train = pd.concat([a,b]).iloc[:,1:-1].fillna(0) #nan 变为0
    X_train = np.array(train,dtype="float32")
    np.random.seed(1)
    shuffle_ix = np.random.permutation(np.arange(len(X_train)))
    data = X_train[shuffle_ix]
    # data = np.array(np.zeros_like(data))
    return data

def Get_DNase_Score(TF_name):
    a = pd.read_table('./data/sequence/'+TF_name+"_DNase.fasta", sep=' ', header=None)
    # a.iloc[:,-1] = a.mean(1)
    b = pd.read_table('./data/sequence/' +TF_name+"_neg_1x_DNase.fasta", sep=' ', header=None)
    train = pd.concat([a,b]).iloc[:,3:-1].fillna(0) #nan 变为0
    # train = preprocessing.scale(train)
    X_train = np.array(train,dtype="float32")
    np.random.seed(1)
    shuffle_ix = np.random.permutation(np.arange(len(X_train)))
    data = X_train[shuffle_ix]
    # data = preprocessing.scale(data)
    # data = np.array(np.zeros_like(data))
    return data

def Get_Histone(TF_name):
    a = pd.read_table('./data/sequence/'+TF_name+"_H3K27me3.fasta", sep=' ', header=None)
    # a.iloc[:,-1] = a.mean(1)
    b = pd.read_table('./data/sequence/' +TF_name+"_neg_1x_H3K27me3.fasta", sep=' ', header=None)
    c = pd.read_table('./data/sequence/' + TF_name + "_H3K9me3.fasta", sep=' ', header=None)
    # a.iloc[:,-1] = a.mean(1)
    d = pd.read_table('./data/sequence/' + TF_name + "_neg_1x_H3K9me3.fasta", sep=' ', header=None)
    train = pd.concat([a,b]).iloc[:,3:-1].fillna(0) #nan 变为0
    train2 = pd.concat([c, d]).iloc[:, 3:-1].fillna(0)
    # train = preprocessing.scale(train)
    X_train = np.array(train,dtype="float32")
    X_train2 = np.array(train2, dtype="float32")
    np.random.seed(1)
    shuffle_ix = np.random.permutation(np.arange(len(X_train)))
    data1 = X_train[shuffle_ix]
    data2 = X_train[shuffle_ix]
    data1 = data1.reshape(data1.shape[0],1,data1.shape[1])
    data2 = data2.reshape(data2.shape[0],1,data2.shape[1])
    data = np.concatenate([data1,data2],1)
    # data = preprocessing.scale(data)
    # data = np.array(np.zeros_like(data))
    return data

def Get_DNA_Sequence(TF_name):
    X_train = []
    y_train = []
    z_train = []
    zero_vector = [0., 0., 0., 0., 0., 0., 0., 0.]
    pos_file = open("./data/sequence/"+TF_name+"_pos.fasta",'r')
    neg_file = open("./data/sequence/"+TF_name+"_neg_1x.fasta",'r')
    sample = []
    pos_num = 0
    neg_num = 0
    length = 101
    number = 0
    # print("READ DNA for %s" % (cell))
    for line in pos_file:

        size = 0
        line = line.strip('\n\r')

        if len(line) < 5:
            break
        if line[0] == ">":
            i = 0
            continue

        else:
            line = line.upper()
            if i == 0:
               content = line
               i = i + 1
               continue
            else:
               content = content+line


        # if number < 511:
        #     number = number + 1
        #     continue

        if len(content) > length:
            size = length
        else:
            size = len(content)
        # row = np.random.randn(101, 4)
        content = 'N' + content + 'N'
        content1 = split_str(content,3 ,101)
        # one_hot_results = tokenizer.texts_to_matrix(content1, mode='binary')
        # one_hot_results = one_hot_results[:,0:125]
        # X_train.append(one_hot_results)
        # for location, base in enumerate(range(0,size), start=0):
        #     row[location] = gene_map[content[base]]
        # X_train.append(row)
        row = np.random.randn(101, 64)
        for location, base in enumerate(range(0, size), start=0):
            row[location] = f[content1[base]]
        X_train.append(row)
        pos_num = pos_num + 1
        y_train.append(1)

    for line in neg_file:

        size = 0
        line = line.strip('\n\r')
        if len(line) < 5:
            break
        if line[0] == ">":
            i = 0
            continue

        else:
            if i == 0:
               content = line
               i = i + 1
               continue
            else:
               content = content + line

        if len(content) > length:
            size = length
        else:
            size = len(content)
        content = 'N' + content + 'N'
        content1 = split_str(content,3 ,101)
        # one_hot_results = tokenizer.texts_to_matrix(content1, mode='binary')
        # one_hot_results = one_hot_results[:,0:125]
        # X_train.append(one_hot_results)
        # row = np.random.randn(101, 4)
        # for location, base in enumerate(range(0,size), start=0):
        #     row[location] = gene_map[content[base]]
        # X_train.append(row)
        row = np.random.randn(101, 64)
        for location, base in enumerate(range(0, size), start=0):
            row[location] = f[content1[base]]
        X_train.append(row)
        neg_num = neg_num + 1
        y_train.append(0)

    print("the number of positive train sample: %d" % pos_num)
    print("the number of negative train sample: %d" % neg_num)
    X_train = np.array(X_train,dtype="float32")
    y_train = np.array(y_train,dtype="float32")
    np.random.seed(1)
    shuffle_ix = np.random.permutation(np.arange(len(X_train)))
    data = X_train[ shuffle_ix ]
    label = y_train[ shuffle_ix ]
    return data, label
# data = Get_Conservation_Score("TF_ATF3_Tissue_Liver")
# x, y = Get_DNA_Sequence("TF_ATF3_Tissue_Liver")
# c
#
# shuffle_ix = np.random.permutation(np.arange(len(x)))
# train_data = x[shuffle_ix]
# train_label = y[shuffle_ix]