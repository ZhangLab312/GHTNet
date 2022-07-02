import numpy as np
import torch
import sys
import pandas as pd
from sklearn import preprocessing
# from keras_preprocessing.text import Tokenizer
import gc
gene_map = {
    'A': [1, 0, 0, 0],
    'C': [0, 1, 0, 0],
    'G': [0, 0, 1, 0],
    'T': [0, 0, 0, 1],
    'N': [0, 0, 0, 0],

}

f = pd.read_csv('./weight_all_32_.csv')
f = f.to_dict('list')


def split_str(s, length, num):
    str = []
    for i in range(num):
        str.append(s[i:i+length])
    return str
def Get_Conservation_Score(TF_name):
    a = pd.read_table('./data/cs/'+TF_name+"_pos_cs.fasta", sep=' ', header=None)
    a.iloc[:,-1] = a.mean(1)
    b = pd.read_table('./data/cs/' +TF_name+"_neg_1x_cs.fasta", sep=' ', header=None)
    train = pd.concat([a,b]).iloc[:,1:-1].fillna(0)
    X_train = np.array(train,dtype="float32")
    np.random.seed(1)
    shuffle_ix = np.random.permutation(np.arange(len(X_train)))
    data = X_train[shuffle_ix]
    return data

def Get_DNase_Score(TF_name):
    a = pd.read_table('./data/dnase/'+TF_name+"_DNase.fasta", sep=' ', header=None)
    b = pd.read_table('./data/dnase/' +TF_name+"_neg_1x_DNase.fasta", sep=' ', header=None)
    train = pd.concat([a,b]).iloc[:,3:-1].fillna(0) #nan 变为0
    X_train = np.array(train,dtype="float32")
    np.random.seed(1)
    shuffle_ix = np.random.permutation(np.arange(len(X_train)))
    data = X_train[shuffle_ix]
    return data

def Get_Histone(TF_name):
    a = pd.read_table('./data/histone/'+TF_name+"_H3K27me3.fasta", sep=' ', header=None)
    b = pd.read_table('./data/histone/' +TF_name+"_neg_1x_H3K27me3.fasta", sep=' ', header=None)
    c = pd.read_table('./data/histone/' + TF_name + "_H3K9me3.fasta", sep=' ', header=None)
    d = pd.read_table('./data/histone/' + TF_name + "_neg_1x_H3K9me3.fasta", sep=' ', header=None)
    train = pd.concat([a,b]).iloc[:,3:-1].fillna(0) #nan 变为0
    train2 = pd.concat([c, d]).iloc[:, 3:-1].fillna(0)
    X_train = np.array(train,dtype="float32")
    X_train2 = np.array(train2, dtype="float32")#(np.zeros_like(X_train))
    np.random.seed(1)
    shuffle_ix = np.random.permutation(np.arange(len(X_train)))
    data1 = X_train[shuffle_ix]
    data2 = X_train2[shuffle_ix]
    data1 = data1.reshape(data1.shape[0],1,data1.shape[1])
    data2 = data2.reshape(data2.shape[0],1,data2.shape[1])
    data = np.concatenate([data1,data2],1)
    return data


def read_shape(TF_Name,Shape):
    with open("./data/se/" + TF_Name + "_pos.data" ,'r') as file:
        train_len = len(file.readlines())
    with open("./data/se/" + TF_Name + "_neg.data" ,'r') as file:
        test_len = len(file.readlines())
    num = len(Shape)
    k = 0
    shape_train = np.random.randn(num, train_len, 101)
    shape_test = np.random.randn(num, test_len, 101)

    for name in Shape:
        train_shape = open("./data/se/" + TF_Name + "_pos_shape.data." + name,'r')
        test_shape = open("./data/se/" + TF_Name + "_neg_shape.data." + name,'r')
        n = 1
        i = 0
        j = 0

        row = np.random.randn(101)
        for line in train_shape:

            line = line.strip('\n\r')
            if line[0] == '>':
                if n != 1:
                    i = 0
                    shape_train[k][j] = row
                    row = np.random.randn(101)
                    if name == "HelT" or name == "Roll"or name == "Rise"or name == "Shift"or name == "Slide"or name == "Roll"or name == "Tilt":

                        row[i] = 0
                        i += 1
                    j += 1
                else:
                    n = 0
                    if name == "HelT" or name == "Roll"or name == "Rise"or name == "Shift"or name == "Slide"or name == "Roll"or name == "Tilt":

                        row[i] = 0
                        i += 1
                continue
            line = line.split(',')
            for s in line:
                if s == 'NA':
                    row[i] = 0
                    i += 1
                else:
                    row[i] = float(s)
                    i += 1
        shape_train[k][j] = row


        n = 1
        i = 0
        j = 0
        row = np.random.randn(101)
        for line in test_shape:

            line = line.strip('\n\r')
            if line[0] == '>':
                if n != 1:
                    i = 0
                    shape_test[k][j] = row
                    row = np.random.randn(101)
                    if name == "HelT" or name == "Roll"or name == "Rise"or name == "Shift"or name == "Slide"or name == "Roll"or name == "Tilt":
                        row[i] = 0
                        i += 1
                    j += 1
                else:
                    n = 0
                    if name == "HelT" or name == "Roll"or name == "Rise"or name == "Shift"or name == "Slide"or name == "Roll"or name == "Tilt":
                        row[i] = 0
                        i += 1
                continue
            line = line.split(',')
            for s in line:
                if s == 'NA':
                    row[i] = 0
                    i += 1
                else:
                    row[i] = float(s)
                    i += 1
        shape_test[k][j] = row
        k += 1
        train_shape.close()
        test_shape.close()

    shape_train = np.append(shape_train, shape_test, 1)
    shape_train = np.transpose(shape_train,(1,0,2))
    np.random.seed(1)
    shuffle_ix = np.random.permutation(np.arange(len(shape_train)))
    data1 = shape_train[shuffle_ix]
    return data1

def Get_DNA_Sequence(TF_name):
    X_train = []
    y_train = []
    pos_file = open("./data/sequence/"+TF_name+"_pos.fasta",'r')
    neg_file = open("./data/sequence/"+TF_name+"_neg_1x.fasta",'r')
    pos_num = 0
    neg_num = 0
    length = 101
    for line in pos_file:
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

        if len(content) > length:
            size = length
        else:
            size = len(content)
        content = 'N' + content + 'N'
        content1 = split_str(content,3 ,101)
        row = np.random.randn(101, 32)
        for location, base in enumerate(range(0, size), start=0):
            row[location] = f[content1[base]]
        X_train.append(row)
        pos_num = pos_num + 1
        y_train.append(1)

    for line in neg_file:
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
        row = np.random.randn(101,32)
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
