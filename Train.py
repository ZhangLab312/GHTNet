import os
import read_data
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import numpy as np
import itertools
import torch.nn as nn
import torch
import pandas as pd
import torch.utils.data as Data
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from model import make_model
from tqdm import tqdm
from sklearn.metrics import accuracy_score

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda")

class EarlyStopping:
    """
        Early stops the training if validation loss doesn't improve after a given patience.
    """

    def __init__(self, patience=5, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float):  Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model1,model2, path):
        """
            Define the __call__ function -> treat a class as a function
        """

        score = -val_loss

        if self.best_score is None:
            """
                Initialization
            """
            self.best_score = score
            self.save_checkpoint(val_loss, model1,model2, path)
        elif score < self.best_score + self.delta:

            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                return 0
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model1,model2, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model1,model2, path):
        """
            Saves model when validation loss decrease.
        """
        if self.verbose:
            print(f'\nValidation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...\n')
        """
            Save Model Parameters
        """
        torch.save(model1.state_dict(), path+"_omics.pth")
        torch.save(model2.state_dict(), path+"_final.pth")
        self.val_loss_min = val_loss

def run(TrainLoader, ValidateLoader,TestLoader, Net, optimizer, loss_function, MAX_EPOCH, lr,net,name):#
    """
        Standard Training
    """

    Net.to(device=device)
    net.to(device=device)
    early_stopping = EarlyStopping(7, True)
    flag = 1
    for epoch in range(MAX_EPOCH):
        if flag == 0:
            break
        Net.train()
        net.train()
        acc = []
        ProgressBar = tqdm(TrainLoader)
        for data in ProgressBar:
            optimizer.zero_grad()
            ProgressBar.set_description("Epoch %d" % epoch)
            samples, label, cs_train,shape_train, histone_train, dnase_train= data#
            data = torch.cat([ cs_train.to(device),dnase_train.to(device), histone_train.to(device),shape_train.to(device)], 1)#
            output_omics = net.forward(src=data.to(device), src_mask=None,data=None)
            output = Net.forward(src=samples.to(device), src_mask=None,data=output_omics)
            loss = loss_function(output.to(device), label.type(torch.LongTensor).to(device))
            """
                Real-time display loss
            """
            ProgressBar.set_postfix(loss=loss.item())
            acc.append(accuracy_score(y_pred=torch.argmax(output.cpu(), 1),
                                      y_true=label.cpu()))
            loss.to(device).backward()
            optimizer.step()
        acc_avg = torch.mean(torch.Tensor(acc))
        for p in optimizer.param_groups:
            if p['lr'] < 0.0005:
                p['lr'] = lr + epoch * 0.00005
        print(f'\nTrain Accuracy: {acc_avg:.3f}\n')

        """
            Validate
        """
        Net.eval()
        net.eval()
        acc = []
        auc_label = []
        auc_score = []
        for valid_samples, valid_labels,cs_vaild,shape_valid,histone_vaild ,  dnase_vaild in ValidateLoader:#
            """
                
            """
            data = torch.cat([ cs_vaild.to(device),dnase_vaild.to(device), histone_vaild.to(device),shape_valid.to(device)], 1)#
            output_omics = net.forward(src=data.to(device), src_mask=None,data=None)
            valid_output = Net.forward(src=valid_samples.to(device), src_mask=None,data=output_omics)
            """
               evaluation
            """
            for k in range(len(valid_labels)):
                auc_label.append(valid_labels.cpu().numpy()[ k ])
                auc_score.append(valid_output.data.cpu().numpy()[ k ][ 1 ])
            valid_labels = valid_labels.to(device)
            acc.append(accuracy_score(y_pred=torch.argmax(valid_output.cpu(), 1),
                                      y_true=valid_labels.cpu()))
        acc_avg = torch.mean(torch.Tensor(acc))
        AUROC = roc_auc_score(auc_label, auc_score)
        precision, recall, thresholds = precision_recall_curve(auc_label, auc_score)
        AUPRC = auc(recall, precision)
        print('Validate auc:%.4f' % AUROC)
        print('Validate prc:%.4f' % AUPRC)
        print(f'\nValidate Accuracy: {acc_avg:.4f}\n')
        """
                early stop
        """
        flag = early_stopping(3-AUPRC-AUROC-acc_avg, net ,Net, './GHTNet/'+name)
    """
        Test
    """
    acc = []
    auc_label = []
    auc_score = []
    Net.load_state_dict(torch.load('./GHTNet/'+name))
    net.load_state_dict(torch.load('./GHTNet/'+name))
    Net.eval()
    net.eval()
    for test_samples, test_labels, cs_test, shape_test, histone_test, dnase_test in TestLoader:  #
        """
            shape [batch_size, NUM_CLASS] 
        """
        data = torch.cat(
            [cs_test.to(device), dnase_test.to(device), histone_test.to(device), shape_test.to(device)], 1)  #
        output_omics = net.forward(src=data.to(device), src_mask=None, data=None)
        test_output = Net.forward(src=test_samples.to(device), src_mask=None, data=output_omics)
        """
            evaluation
        """
        for k in range(len(test_labels)):
            auc_label.append(test_labels.cpu().numpy()[k])
            auc_score.append(test_output.data.cpu().numpy()[k][1])
        test_labels = test_labels.to(device)
        acc.append(accuracy_score(y_pred=torch.argmax(test_output.cpu(), 1),
                                  y_true=test_labels.cpu()))
    acc_avg = torch.mean(torch.Tensor(acc))
    AUROC = roc_auc_score(auc_label, auc_score)
    precision, recall, thresholds = precision_recall_curve(auc_label, auc_score)
    AUPRC = auc(recall, precision)
    print('Test auc:%.4f' % AUROC)
    print('Test prc:%.4f' % AUPRC)
    print(f'\nTest Accuracy: {acc_avg:.4f}\n')
    """
            save result
    """
    file = open('./GHTNet.txt', "a")
    file.write(name + " " + str(np.round(AUROC, 4)) + " " + str(np.round(acc_avg, 4)) + " " + str(np.round(AUPRC, 4)) +"\n")
    file.close()
    print('\n---Leaving  Train  Section---\n')


"""
   load_data
"""
def load_data(name):
    shape = read_data.read_shape(name[0:-10],[ "ProT","Stretch","Buckle","Shear",  "Opening","Stagger","Rise", "Shift", "Slide","MGW","HelT", "Roll",  "Tilt","EP"])

    shape = shape.astype(np.float32)

    X, Y = read_data.Get_DNA_Sequence(name[0:-10])

    cs = read_data.Get_Conservation_Score(name[0:-10]).astype(np.float32)


    DNase = read_data.Get_DNase_Score(name[0:-10])
    histone = read_data.Get_Histone(name[0:-10])
    cs = cs[:, np.newaxis,: ]
    DNase = DNase[:, np.newaxis,: ]

    number = len(X) // 10

    shape_train = shape[0 * number:8 * number]
    shape_test = shape[9 * number:10 * number]
    shape_validation = shape[8 * number:9 * number]

    cs_train = cs[0 * number:8 * number]
    cs_test = cs[9 * number:10 * number]
    cs_validation = cs[8 * number:9 * number]

    histone_train = histone[0 * number:8 * number]
    histone_test = histone[9 * number:10 * number]
    histone_validation = histone[8 * number:9 * number]

    DNase_train = DNase[0 * number:8 * number]
    DNase_test = DNase[9 * number:10 * number]
    DNase_validation = DNase[8 * number:9 * number]

    X_train = X[0 * number:8 * number]
    Y_train = Y[0 * number:8 * number]
    X_test = X[9 * number:10 * number]
    Y_test = Y[9 * number:10 * number]
    X_validation = X[8 * number:9 * number]
    Y_validation = Y[8 * number:9 * number]
    X_test = torch.from_numpy(X_test)
    Y_test = torch.from_numpy(Y_test)
    X_train = torch.from_numpy(X_train)
    Y_train = torch.from_numpy(Y_train)
    X_validation = torch.from_numpy(X_validation)
    Y_validation = torch.from_numpy(Y_validation)
    X_test = torch.transpose(X_test, 1, 2)
    X_train = torch.transpose(X_train, 1, 2)
    X_validation = torch.transpose(X_validation, 1, 2)

    shape_train = torch.from_numpy(shape_train)
    shape_test = torch.from_numpy(shape_test)
    shape_validation = torch.from_numpy(shape_validation)

    cs_train = torch.from_numpy(cs_train)
    cs_test = torch.from_numpy(cs_test)
    cs_validation = torch.from_numpy(cs_validation)

    histone_train = torch.from_numpy(histone_train)
    histone_test = torch.from_numpy(histone_test)
    histone_validation = torch.from_numpy(histone_validation)

    DNase_train = torch.from_numpy(DNase_train)
    DNase_test = torch.from_numpy(DNase_test)
    DNase_validation = torch.from_numpy(DNase_validation)

    TrainLoader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(X_train, Y_train, cs_train, shape_train,histone_train,DNase_train),#
                                              batch_size=64, shuffle=True, num_workers=0)
    TestLoader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(X_test, Y_test ,cs_test,shape_test, histone_test,DNase_test),#
                                             batch_size=64, shuffle=True, num_workers=0 )
    ValidationLoader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(X_validation, Y_validation, cs_validation,shape_validation,histone_validation,DNase_validation),#
                                                   batch_size=64, shuffle=True, num_workers=0)
    return TrainLoader,ValidationLoader,TestLoader


"""
    ***Main***
"""
if __name__=='__main__':
    files = os.listdir('./data/sequence')
    lr = 0.0001
    Net = make_model( NConv=1, NTrans=1,
                     d_model=16, d_ff=32, h=8, dropout=0.2, ms_num=16, )
    net = make_model( NConv=1, NTrans=1,
                     d_model=18, d_ff=36, h=6, dropout=0.2, ms_num=18, )

    optimizer = optim.AdamW(itertools.chain(Net.parameters(), net.parameters()), weight_decay=0.01, betas=(0.9, 0.999),
                            lr=lr)

    criterion = nn.CrossEntropyLoss()

    """
        Training start 
    """
    for name in files:
        if name[-12:] == 'neg_1x.fasta':
             continue
        TrainLoader, ValidationLoader, TestLoader = load_data(name)
        run(TrainLoader=TrainLoader, ValidateLoader=ValidationLoader, TestLoader=TestLoader,
            Net=Net, optimizer=optimizer, loss_function=criterion,net=net,
            MAX_EPOCH=MAX_EPOCH, lr=lr,name=name[0:-10])
