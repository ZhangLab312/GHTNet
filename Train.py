import os
import read_data
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from model import make_model
import torch.nn as nn
import torch
import torch.utils.data as Data
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.metrics import auc
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda")


class Batch:
    """
        To manege a batch of samples with mask in learning process
    """

    def __init__(self, src, pad=0):
        """
            src 是 model 的 input
            [batch_size, max_length, embedding(d_model)]
        """
        self.src = src
        """
            unsqueeze(-2) 在倒数第二个维度 插入一个维度
            e.g  [30, 10] --> [30, 1, 10]
            (src != pad) 如果scr中的元素和pad不相等 则src_mask这个位置是1 如果相等则是0
        """
        self.src_mask = (src != pad).unsqueeze(-2)


class EarlyStopping:
    """
        Early stops the training if validation loss doesn't improve after a given patience.
    """

    def __init__(self, patience=3, verbose=False, delta=0):
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

    def __call__(self, val_loss, model1, model2, path, num):
        """
            定义 __call__ 函数 -> 将一个类视作一个函数
            该函数的目的 类似在class中重载()运算符
            使得这个类的实例对象可以和普通函数一样 call
            即，通过 对象名() 的形式使用
        """

        score = -val_loss

        if self.best_score is None:
            """
                初始化（第一次call EarlyStopping）
                保存检查点
            """
            self.best_score = score
            self.save_checkpoint(val_loss, model1, model2, path, num)
        elif score < self.best_score + self.delta:
            """
                验证集损失没有继续下降时，计数
                当计数 大于 耐心值时，停止
                注：
                    由于模型性能没有改善，此时是不保存检查点的
            """
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                return 0
        else:
            """
                验证集损失下降了，此时从头开始计数
                保存检查点
            """
            self.best_score = score
            self.save_checkpoint(val_loss, model1, model2, path, num)
            self.counter = 0

    def save_checkpoint(self, val_loss, model1, model2, path, num):
        """
            Saves model when validation loss decrease.
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        """
            保存最优的模型Parameters
        """
        torch.save(model1, path + f"_omics_{num}.pth")
        torch.save(model2, path + f"_final_{num}.pth")
        self.val_loss_min = val_loss


def run(train_loader, validate_loader, Net, optimizer, loss_function, max_epoch, lr, net, name, num):  #
    """
        Standard Training
    """
    Net.to(device=device)
    net.to(device=device)
    early_stopping = EarlyStopping(7, True)
    flag = 1
    average_score = 3
    for epoch in range(max_epoch):
        if flag == 0:
            break
        Net.train()
        net.train()
        acc = []
        progress_bar = tqdm(train_loader)
        for data in progress_bar:
            optimizer.zero_grad()
            progress_bar.set_description("Epoch %d" % epoch)
            samples, label, cs_train, shape_train, histone_train, dnase_train = data
            data = torch.cat(
                [cs_train.to(device), dnase_train.to(device), histone_train.to(device), shape_train.to(device)], 1)  #
            output_omics = net.forward(src=data.to(device), src_mask=None, data=None)
            output = Net.forward(src=samples.to(device), src_mask=None, data=output_omics)
            loss = loss_function(output.to(device), label.type(torch.LongTensor).to(device))
            """
                实时显示loss
            """
            progress_bar.set_postfix(loss=loss.item())
            acc.append(accuracy_score(y_pred=torch.argmax(output.cpu(), 1),
                                      y_true=label.cpu()))
            loss.to(device).backward()
            optimizer.step()
        acc_avg = torch.mean(torch.Tensor(acc))
        for p in optimizer.param_groups:
            if p['lr'] < 0.0004:
                p['lr'] = lr + epoch * 0.00003
        print(f'\nTrain Accuracy: {acc_avg:.3f}')
        """
            Validate
        """
        Net.eval()
        net.eval()
        """
                评估指标：Accuracy
            """
        acc = []
        auc_label = []
        auc_score = []
        num1 = 0
        for valid_samples, valid_labels, cs_vaild, shape_valid, histone_vaild, dnase_vaild in validate_loader:
            num1 += 1 
            """
                shape [batch_size, NUM_CLASS] 
            """
            data = torch.cat(
                [cs_vaild.to(device), dnase_vaild.to(device), histone_vaild.to(device), shape_valid.to(device)], 1)
            output_omics = net.forward(src=data.to(device), src_mask=None, data=None)
            valid_output = Net.forward(src=valid_samples.to(device), src_mask=None, data=output_omics)
            """
                multi-class  评估
            """
            with torch.no_grad():
                loss = loss_function(output.to(device), label.type(torch.LongTensor).to(device))
                running_loss += loss.item()
            for k in range(len(valid_labels)):
                auc_label.append(valid_labels.cpu().numpy()[k])
                auc_score.append(valid_output.data.cpu().numpy()[k][1])
            valid_labels = valid_labels.to(device)
            acc.append(accuracy_score(y_pred=torch.argmax(valid_output.cpu(), 1),
                                      y_true=valid_labels.cpu()))
        acc_avg = torch.mean(torch.Tensor(acc))
        auroc = roc_auc_score(auc_label, auc_score)
        precision, recall, thresholds = precision_recall_curve(auc_label, auc_score)
        auprc = auc(recall, precision)
        print('auc:%.4f' % auroc)
        print('prc:%.4f' % auprc)
        print(f'Validate Accuracy: {acc_avg:.4f}')

        """
                保存模型
        """
        if epoch == 0:
            acc_temp = acc_avg
            auc_temp = auroc
            prc_temp = auprc
            average_score = (3 - auprc - auroc - acc_avg)
        else:
            if average_score > (3 - auprc - auroc - acc_avg):
                acc_temp = acc_avg
                auc_temp = auroc
                prc_temp = auprc
                average_score = (3 - auprc - auroc - acc_avg)
        flag = early_stopping(running_loss / num1, net, Net, './module/' + name, num)
    file = open('./module.txt', "a")
    file.write(name + " " + str(np.round(auc_temp, 4)) + " " + str(np.round(acc_temp, 4)) + " " + str(
        np.round(prc_temp, 4)) + "\n")
    file.close()
    print('\n---Leaving  Train  Section---\n')


def run_trained_model(test_loader, net, Net):
    Net.eval()
    net.eval()
    acc = []
    auc_label = []
    auc_score = []
    for samples, labels, cs, shape, histone, dnase in test_loader:
        """
            shape [batch_size, NUM_CLASS] 
        """
        data = torch.cat(
            [cs.to(device), dnase.to(device), histone.to(device), shape.to(device)], 1)
        output_omics = net.forward(src=data.to(device), src_mask=None, data=None)
        output = Net.forward(src=samples.to(device), src_mask=None, data=output_omics)
        """
            multi-class  评估
        """
        for k in range(len(labels)):
            auc_label.append(labels.cpu().numpy()[k])
            auc_score.append(output.data.cpu().numpy()[k][1])
        labels = labels.to(device)
        acc.append(accuracy_score(y_pred=torch.argmax(output.cpu(), 1),
                                  y_true=labels.cpu()))
    acc_avg = torch.mean(torch.Tensor(acc))
    auroc = roc_auc_score(auc_label, auc_score)
    precision, recall, thresholds = precision_recall_curve(auc_label, auc_score)
    auprc = auc(recall, precision)

    return round(float(acc_avg), 4), round(auroc, 4), round(auprc, 4)


def load_data(name):
    shape = read_data.read_shape(name[0:-10],
                                 ["ProT", "Stretch", "Buckle", "Shear", "Opening", "Stagger", "Rise", "Shift", "Slide",
                                  "MGW", "HelT", "Roll", "Tilt", "EP"])
    shape = shape.astype(np.float32)
    X, Y = read_data.Get_DNA_Sequence(name[0:-10])
    cs = read_data.Get_Conservation_Score(name[0:-10])
    histone = read_data.Get_Histone(name[0:-10])
    cs = cs[:, np.newaxis, :]
    DNase = read_data.Get_DNase_Score(name[0:-10])
    DNase = DNase[:, np.newaxis, :]
    number = len(X) // 10
    shape_kf = torch.from_numpy(shape[: 9*number])
    shape_test = torch.from_numpy(shape[9*number: 10*number])

    X_kf = torch.from_numpy(X[: 9*number])
    X_kf = torch.transpose(X_kf, 1, 2)
    X_test = torch.from_numpy(X[9*number: 10*number])
    X_test = torch.transpose(X_test, 1, 2)

    Y_kf = torch.from_numpy(Y[: 9*number])
    Y_test = torch.from_numpy(Y[9*number: 10*number])

    cs_kf = torch.from_numpy(cs[: 9*number])
    cs_test = torch.from_numpy(cs[9*number: 10*number])

    histone_kf = torch.from_numpy(histone[: 9*number])
    histone_test = torch.from_numpy(histone[9*number: 10*number])

    DNase_kf = torch.from_numpy(DNase[: 9*number])
    DNase_test = torch.from_numpy(DNase[9*number: 10*number])

    test_loader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(X_test, Y_test, cs_test, shape_test, histone_test, DNase_test),
        batch_size=64, shuffle=True, num_workers=0
    )
    kf = TensorDataset(X_kf, Y_kf, cs_kf, shape_kf, histone_kf, DNase_kf)
    return kf, test_loader


max_epoch = 25
num_class = 2
files = os.listdir('./data/sequence')
lr = 0.0005
path = os.getcwd() + '/trans32.pth'
criterion = nn.CrossEntropyLoss()


if __name__ == '__main__':
    """
        Training start 
    """
    for name in files:
        if name[-12:] == 'neg_1x.fasta':
            continue
        kf_dataset, test_loader = load_data(name)
        kf = KFold(n_splits=5, shuffle=True)
        num = 0
        for train_idx, validate_idx in kf.split(kf_dataset):
            num += 1
            Net = make_model(TFs_cell_line_pair=num_class, NConv=1, NTrans=1,
                             d_model=16, d_ff=32, h=2, dropout=0.2, ms_num=16, )
            net = make_model(TFs_cell_line_pair=num_class, NConv=1, NTrans=1,
                             d_model=18, d_ff=36, h=6, dropout=0.2, ms_num=18, )
            optimizer = optim.AdamW(itertools.chain(Net.parameters(), net.parameters()), weight_decay=0.01,
                                    betas=(0.9, 0.999),
                                    lr=lr)
            train_data = Subset(kf_dataset, train_idx)
            validate_data = Subset(kf_dataset, validate_idx)
            train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
            validate_loader = DataLoader(validate_data, batch_size=64, shuffle=True)
            run(train_loader=train_loader, validate_loader=validate_loader,
                Net=Net, optimizer=optimizer, loss_function=criterion, net=net,
                max_epoch=max_epoch, lr=lr, name=name[0:-10], num=num)
        evaluation = []
        name = name[0: -10]
        for i in range(5):
            net_path = './module/' + name + f"_omics_{i + 1}.pth"
            Net_path = './module/' + name + f"_final_{i + 1}.pth"
            net = torch.load(net_path)
            Net = torch.load(Net_path)
            acc, aur, aup = run_trained_model(test_loader, net, Net)
            evaluation.append({
                "acc": acc,
                "aur": aur,
                "aup": aup
            })
        df = pd.DataFrame(evaluation)
        avg = pd.DataFrame(df.mean()).T
        avg.index = ['avg']
        df = pd.concat([df, avg])
        df.to_csv('./evaluation.csv')
