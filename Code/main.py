import random
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from load_data import *
from utils import *
from stgcn import *
import time
import matplotlib.pyplot as plt

#Random Seed
torch.manual_seed(1000)
torch.cuda.manual_seed(1000)
np.random.seed(1000)
random.seed(1000)

#Hardware Setting
torch.backends.cudnn.deterministic = True
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#File path
matrix_path = "dataset/weighted_adjacency.csv"
data_path = "dataset/historical.csv"
save_path = "save/RES_8.pt"
plot_path = "plot/"

#Model Parameter
'''
day_slot: devide a day to 288 slots which means the data time interval is 5 mins
n_train,n_val,n_test: 34 days data for training, 5 days data for validation and testing
n_his: 12 data slots as input series (60 mins)
n_pred: output 3 data slots on next 12 slots(15,30,45 mins)
n_route: In PeMSD7 data set the adjacency matrix contains 228 nodes
Ks: Kernel of spatio conv
Kt: Kernel of temporal conv
blocks: conv shape
drop_prob: drop out probability
'''
day_slot = 288
n_train, n_val, n_test = 34, 5, 5
n_his = 12
n_pred = 9
n_route = 228
Ks, Kt = 3 , 3
blocks = [[1, 32, 64], [64, 32, 128]]
drop_prob = 0
batch_size = 50
epochs = 100
lr = 1e-3

#ChebyNet Polynomial
W = load_matrix(matrix_path)
L = scaled_laplacian(W)
Lk = cheb_poly(L, Ks)
Lk = torch.Tensor(Lk.astype(np.float32)).to(device)

#Load Data
train, val, test = load_data(data_path, n_train * day_slot, n_val * day_slot)
scaler = StandardScaler()
train = scaler.fit_transform(train)
val = scaler.transform(val)
test = scaler.transform(test)

x_train, y_train = data_transform(train, n_his, n_pred, day_slot, device)
x_val, y_val = data_transform(val, n_his, n_pred, day_slot, device)
x_test, y_test = data_transform(test, n_his, n_pred, day_slot, device)

train_data = torch.utils.data.TensorDataset(x_train, y_train)
train_iter = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)
val_data = torch.utils.data.TensorDataset(x_val, y_val)
val_iter = torch.utils.data.DataLoader(val_data, batch_size)
test_data = torch.utils.data.TensorDataset(x_test, y_test)
test_iter = torch.utils.data.DataLoader(test_data, batch_size)

#Model construct
loss = nn.MSELoss()
model = STGCN(Ks, Kt, blocks, n_his, n_route, Lk, drop_prob)
model =torch.nn.DataParallel(model)
model.to(device)
optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

train_losses = []
val_losses = []

def train():
    tStart = time.time()
    min_val_loss = np.inf
    for epoch in range(1, epochs + 1):
        l_sum, n = 0.0, 0
        model.module.train()
        for x, y in train_iter:
            y_pred = model(x).view(len(x), -1)
            l = loss(y_pred, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        scheduler.step()
        val_loss = evaluate_model(model, loss, val_iter)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.module.state_dict(), save_path)
        train_losses.append(l_sum/n)
        val_losses.append(val_loss)
        print("epoch", epoch, ", train loss:", l_sum / n, ", validation loss:", val_loss)
    tEnd = time.time()
    print("\nTime:%f" %(tEnd-tStart))
    
def evaluate():
    best_model = STGCN(Ks, Kt, blocks, n_his, n_route, Lk, drop_prob).to(device)
    best_model.load_state_dict(torch.load('./save/RES_8.pt'))    
    l = evaluate_model(best_model, loss, test_iter)
    MAE, MAPE, RMSE = evaluate_metric(best_model, test_iter, scaler)
    print("test loss:", l, "\nMAE:", MAE, ", MAPE:", MAPE, ", RMSE:", RMSE)


def plot1():
    x = range(0,100)
    y1t = np.load('./plot/Res8_train_loss.npy')
    y1v = np.load('./plot/Res8_val_train_loss.npy')
    y2t = np.load('./plot/Res16_train_loss.npy')
    y2v = np.load('./plot/Res16_val_train_loss.npy')
    y3t = np.load('./plot/Res32_train_loss.npy')
    y3v = np.load('./plot/Res32_val_train_loss.npy')
    y4t = np.load('./plot/Res56_train_loss.npy')
    y4v = np.load('./plot/Res56_val_train_loss.npy')
    plt.plot(x, y1v, '.-r',label='RES_8')
    plt.plot(x, y2v, '.-g',label='RES_16')
    plt.plot(x, y3v, '.-b',label='RES_32')
    plt.plot(x, y4v, '.-y',label='RES_56')
    plt.xlabel('validation Loss of deep ResGCN')
    # plt.plot(x, y1t, '.-r',label='RES_8')
    # plt.plot(x, y2t, '.-g',label='RES_16')
    # plt.plot(x, y3t, '.-b',label='RES_32')
    # plt.plot(x, y4t, '.-y',label='RES_56')
    # plt.xlabel('train Loss of deep ResGCN')
    plt.legend(loc='best')
    plt.show()
    
def plot2():
    x = range(0,100)
    
    y1t = np.load('./plot/STGCN2_train_loss.npy')
    y1v = np.load('./plot/STGCN2_val_train_loss.npy')
    y2t = np.load('./plot/STGCN8_train_loss.npy')
    y2v = np.load('./plot/STGCN8_val_train_loss.npy')
    y3t = np.load('./plot/STGCN16_train_loss.npy')
    y3v = np.load('./plot/STGCN16_val_train_loss.npy')
    y4t = np.load('./plot/STGCN32_train.loss.npy')
    y4v = np.load('./plot/STGCN32_val_train_loss.npy')
    y5t = np.load('./plot/STGCN56_train_loss.npy')
    y5v = np.load('./plot/STGCN56_val_train_loss.npy')
    plt.plot(x, y1v, '.-r',label='STGCN_2')
    plt.plot(x, y2v, '.-g',label='STGCN_8')
    plt.plot(x, y3v, '.-b',label='STGCN_16')
    plt.plot(x, y4v, '.-y',label='STGCN_32')
    plt.plot(x, y5v, '.-c',label='STGCN_56')
    plt.legend(loc='best')
    plt.xlabel('validation Loss of deep STGCN')
    plt.plot(x, y1t, '.-r',label='STGCN_8')
    plt.plot(x, y2t, '.-g',label='STGCN_16')
    plt.plot(x, y3t, '.-b',label='STGCN_32')
    plt.plot(x, y4t, '.-y',label='STGCN_56')
    plt.xlabel('train Loss of deep STGCN')
    
    plt.show()
    
train()
# np.save(plot_path+'train_loss.npy',train_losses)
# np.save(plot_path+'val_train_loss.npy',val_losses)
evaluate()
plot1()
plot2()
