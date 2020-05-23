import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
from . import config, utils


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data, 0)
    if isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight.data, 1)
        init.constant_(m.bias.data, 0)

class BasicWideBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout_p=.5):
        super(BasicWideBlock, self).__init__()
        
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout2d(dropout_p)
        
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding=1)
        
        if in_channels != out_channels or stride != 1:
            self.shunt = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=stride)
        else:
            self.shunt = nn.Identity()
    
    def forward(self, x):
        # x: (batch_size, in_channels, H, W)
        
        # first convolution
        out = self.conv1(self.activation(self.bn1(x)))
        
        # dropout
        out = self.dropout(out)
        
        # second convolution
        out = self.conv2(self.activation(self.bn2(out)))
        
        # return residual
        out += self.shunt(x)
        
        # out: (batch_size, out_channels, (H - kernel_size - 1) / stride + 1, (W - kernel_size - 1) / stride + 1)
        return out


class WideResNet(nn.Module):
    '''Wide ResNet architecture with an arbitrary number of stages as described in the original paper.
    '''
    def __init__(self, in_channels, first_out_channels, n_classes, k, n, n_stages=3, dropout_p=.5):
        super(WideResNet, self).__init__()
        
        # first layer is a convolution
        self.conv1 = nn.Conv2d(in_channels, first_out_channels, kernel_size=3, padding=1)
        self._in_channels = first_out_channels
        
        # build network stages
        stages = []
        stages.append(self._make_stage(n, first_out_channels*k, kernel_size=3, stride=1, dropout_p=dropout_p))
        
        for i in range(1, n_stages):
            n_out_channels = first_out_channels*(2**i)*k
            stages.append(self._make_stage(n, n_out_channels, kernel_size=3, stride=2, dropout_p=dropout_p))
        
        self.stages = nn.Sequential(*stages)
        
        self.bn = nn.BatchNorm2d(n_out_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.activation = nn.LeakyReLU()
        self.linear = nn.Linear(n_out_channels, n_classes)
    
    def _make_stage(self, n, out_channels, kernel_size, stride, dropout_p):
        # reduce (H,W) with the stride on the first layer only
        strides = np.ones(n, dtype=np.int16)
        strides[0] = stride
        blocks = []
        
        for i, stride in enumerate(strides):
            blocks.append(BasicWideBlock(self._in_channels, out_channels, kernel_size=kernel_size, stride=stride, dropout_p=dropout_p))
            
            # update the number of channels once for the rest of the blocks
            if i == 0:
                self._in_channels = out_channels
            
        return nn.Sequential(*blocks)
    
    def forward(self, x):
        
        # first convolution
        out = self.conv1(x)
        
        # pass through stages
        out = self.stages(out)
        
        # bn and activation
        out = self.activation(self.bn(out))
        
        # pool, reshape and pass through dense
        out = self.pool(out)
        out = torch.squeeze(out)
        out = self.linear(out)
        
        return out


def train(model, criterion, optimizer, dl_train, dl_val, n_epochs, restart_epoch, model_name, min_epoch=10, patience=5, verbose=1):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    loss_history = []
    eval_loss_history = []
    eval_roc_auc_history = []
    best_eval_roc_auc = 0
    trials = 0
    
    for epoch in range(restart_epoch, n_epochs + 1):
        # perform a full pass on the dataset
        model.train()
        epoch_loss = 0

        for i, batch in enumerate(dl_train):
            X, y = [ds.to(device) for ds in batch]

            # erase gradients and perform a forward pass
            optimizer.zero_grad()
            y_hat = model(X)
            # compute loss and perform a backward pass
            loss = criterion(y_hat, torch.argmax(y, dim=1))
            
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

            del X, y
            torch.cuda.empty_cache()

        epoch_loss /= len(dl_train.dataset)
        loss_history.append(epoch_loss)
        
        # evaluate
        eval_loss, eval_roc_auc, _, _ = evaluate(model, criterion, optimizer, dl_val)
        eval_loss_history.append(eval_loss)
        eval_roc_auc_history.append(eval_roc_auc)

        if epoch % verbose == 0:
            print(f'Epoch: {epoch:3d} - Train loss: {epoch_loss:.4f} | Eval loss: {eval_loss:.4f} - Eval ROC-AUC: {eval_roc_auc:.4f}')

            if epoch >= min_epoch:
                if eval_roc_auc > best_eval_roc_auc:
                    trials = 0
                    best_eval_roc_auc = eval_roc_auc
                    
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, config.get_model_filename(model_name))

                else:
                    trials += 1
                    if trials >= patience:
                        print(f'Early stopping on epoch {epoch}')
                        break
    
    return loss_history, eval_loss_history, eval_roc_auc_history


def evaluate(model, criterion, optimizer, dl_val):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    model.eval()
    eval_loss = 0
    
    with torch.no_grad():
        y_val_hat_running = torch.tensor([], device=device)
        y_val_running = torch.tensor([], dtype=torch.long, device=device)
        
        for batch in dl_val:
            X, y = [ds.to(device) for ds in batch]
            y_hat = model(X)

            loss = criterion(y_hat, torch.argmax(y, dim=1))
            eval_loss += loss.item()
            
            y_hat = torch.softmax(y_hat, dim=1)
            
            y_val_hat_running = torch.cat([y_val_hat_running, y_hat], 0)
            y_val_running = torch.cat([y_val_running, y], 0)

            del X, y
            torch.cuda.empty_cache()
            
    y_val_hat_running = y_val_hat_running.cpu().numpy()
    y_val_running = y_val_running.cpu().numpy()
    
    eval_loss /= len(dl_val.dataset)
    eval_roc_auc = utils.mean_column_wise_roc_auc(y_val_hat_running, y_val_running)
    
    return eval_loss, eval_roc_auc, y_val_hat_running, y_val_running


def predict(model, dl_test):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    model.eval()
    y_hat_running = torch.tensor([], device=device)
    
    with torch.no_grad():
        for batch in dl_test:
            X = [ds.to(device) for ds in batch]
            y_hat = model(X)            
            y_hat = torch.softmax(y_hat, dim=1)
            y_hat_running = torch.cat([y_val_hat_running, y_hat], 0)
            
            del X
            torch.cuda.empty_cache()
    
    return y_hat_running.cpu().numpy()