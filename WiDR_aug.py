import argparse
import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
import scipy.io as sio
import torchcde
import torch.nn.functional as F
from utils.mixup import mixup

# Argument parsing
parser = argparse.ArgumentParser(description="Run Neural CDE Training")
parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--FFN', type=int, default=256, help='Feedforward network')
parser.add_argument('--MHA', type=int, default=4, help='Multihead attention')
parser.add_argument('--nnn', type=int, default=400, help='Neural CDE hidden nodes')
parser.add_argument('--sharing', type=float, default=0.7, help='parameter sharing')
parser.add_argument('--alpha', type=float, default=0.7, help='Data augmentation strength')
args = parser.parse_args()



# GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Load train data
data_amp = sio.loadmat('./data/ARIL/train_data_split_amp.mat')
train_data_amp = data_amp['train_data']
train_data = torch.from_numpy(train_data_amp).type(torch.FloatTensor).transpose(1, 2)
train_label = torch.from_numpy(data_amp['train_activity_label']).type(torch.LongTensor).squeeze()
train_label1 = torch.from_numpy(data_amp['train_location_label']).type(torch.LongTensor).squeeze()

# Load test data
data_amp = sio.loadmat('./data/ARIL/test_data_split_amp.mat')
test_data_amp = data_amp['test_data']
test_data = torch.from_numpy(test_data_amp).type(torch.FloatTensor).transpose(1, 2)
test_label = torch.from_numpy(data_amp['test_activity_label']).type(torch.LongTensor).squeeze()
test_label1 = torch.from_numpy(data_amp['test_location_label']).type(torch.LongTensor).squeeze()

# Z-Score normalization
mean, std = train_data.mean(dim=0, keepdim=True), train_data.std(dim=0, keepdim=True)
train_data, test_data = (train_data - mean) / std, (test_data - mean) / std

# Custom Dataset
class CustomTensorDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels_A, self.labels_B = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, self.labels_A[idx], self.labels_B[idx]

# Neural CDE and helper classes
# (Define PositionalEncoding, BottleneckBlock, CrossAttention, TransformerEncoder, CDEFunc, NeuralCDE)
class CDEFunc(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, nodes):
        super(CDEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.linear1 = torch.nn.Linear(hidden_channels, nodes)
        self.linear2 = torch.nn.Linear(nodes, input_channels * hidden_channels)
    def forward(self, t, z):
        z = self.linear1(z)
        z = z.relu()
        z = self.linear2(z) 
        z = z.tanh()
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
        return z

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x) 
    
    
class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, bottleneck_channels):
        super(BottleneckBlock, self).__init__()
        
        # 1x1 Convolution: Reduce channels
        self.conv_reduce = nn.Conv1d(in_channels, 192, kernel_size=1, stride=1)
        self.bn_reduce = nn.BatchNorm1d(192)
        
        # 3x3 Convolution: Main feature extraction without changing feature size
        self.conv_3x3 = nn.Conv1d(192, 192, kernel_size=3, stride=1, padding=1)
        self.bn_3x3 = nn.BatchNorm1d(192)
        
        # 1x1 Convolution: Reduce channels to 1
        self.conv_reduce2 = nn.Conv1d(192, bottleneck_channels, kernel_size=1, stride=1)
        self.bn_reduce2 = nn.BatchNorm1d(bottleneck_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = x.float()
        
        # 1x1 Convolution
        out = self.conv_reduce(x)
        out = self.bn_reduce(out)
        out = self.relu(out)       
        # 3x3 Convolution
        out = self.conv_3x3(out)
        out = self.bn_3x3(out)
        out = self.relu(out)
        # 1x1 Convolution
        out = self.conv_reduce2(out)
        out = self.bn_reduce2(out)
        out = self.relu(out)
        # Skip connection 추가
        #out += x
        #out = self.relu(out)
        return out    
        
        
class CrossAttention(nn.Module):
    def __init__(self, embed_dim_query, embed_dim_key_value, embed_dim_output, num_heads, ffnn_dim):
        super(CrossAttention, self).__init__()
        self.query_linear = nn.Linear(embed_dim_query, embed_dim_output)
        self.key_value_linear = nn.Linear(embed_dim_key_value, embed_dim_output)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim_output, num_heads=num_heads, batch_first=True)
        
        self.ffnn = nn.Sequential(
            nn.Linear(embed_dim_output, ffnn_dim),
            nn.ReLU(),
            nn.Linear(ffnn_dim, embed_dim_output)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim_output)
        self.norm2 = nn.LayerNorm(embed_dim_output)
        

    def forward(self, query, key, value):
        query_transformed = self.query_linear(query)
        key_transformed = self.key_value_linear(key)
        value_transformed = self.key_value_linear(value)
        attn_output, attn_output_weights = self.multihead_attn(query_transformed, key_transformed, value_transformed)
        
        #Add & Norm 대신 Add만 norm은 입력에
        attn_output = (attn_output + query_transformed)
        attn_output= self.norm2(attn_output)
        ffnn_output = self.ffnn(attn_output)
        output = (ffnn_output + attn_output)
        return output
       
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)            
        self.positional_encoding = PositionalEncoding(d_model)
    def forward(self, src):
        src = self.positional_encoding(src)
        return self.transformer_encoder(src)
class NeuralCDE(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels,nodes, output_channels,Multihead, feedforward, interpolation="cubic"):
        super(NeuralCDE, self).__init__()
        self.func = CDEFunc(input_channels, hidden_channels, nodes)
        self.initial = torch.nn.Linear(input_channels, hidden_channels)
        self.readout = torch.nn.Linear(hidden_channels, output_channels)
        self.interpolation = interpolation
        
        self.norm = nn.LayerNorm(52)
        self.cross_attn = CrossAttention(embed_dim_query=192, embed_dim_key_value=52, embed_dim_output=192, num_heads=Multihead,ffnn_dim=feedforward) #default = 4, 1024
        self.cross_attn1 = CrossAttention(embed_dim_query=192, embed_dim_key_value=52, embed_dim_output=192, num_heads=Multihead,ffnn_dim=feedforward) #default = 4, 1024
        
        
        self.fc = nn.Linear(192*52, 200)
        self.norm3 = nn.LayerNorm(200)
        self.fc1 = nn.Linear(200, 6)
        
        self.fc2 = nn.Linear(192*52, 200)
        self.norm4 = nn.LayerNorm(200)
        self.fc3 = nn.Linear(200, 16)

    def forward(self, coeffs):
        if self.interpolation == 'cubic':
            X = torchcde.CubicSpline(coeffs)
        elif self.interpolation == 'linear':
            X = torchcde.LinearInterpolation(coeffs)
        else:
            raise ValueError("Only 'linear' and 'cubic' interpolation methods are implemented.")

        X0 = X.evaluate(X.interval[0])
        z0 = self.initial(X0)
        z_T = torchcde.cdeint(X=X,
                              z0=z0,
                              func=self.func,
                              t=X.grid_points)

        z_T = self.norm(z_T)
        z_T = F.relu(z_T)
        
        z_TT = z_T.permute(0,2,1)
        
        pred_y = self.cross_attn(query =z_TT, key = z_T, value = z_T)
        pred_y1 = self.cross_attn1(query =z_TT, key = z_T, value = z_T)

        
        temp = (pred_y.shape)
        temp1 = (pred_y1.shape)
        
        
        pred_y = pred_y.reshape(temp[0],-1)
        pred_y1 = pred_y1.reshape(temp1[0],-1)
        
        pred_y11 = (self.fc(pred_y))
        pred_y11 = self.norm3(pred_y11)
        pred_y11 = F.relu(pred_y11)      
        pred_y11 = (self.fc1(pred_y11))
        
        pred_y2 = (self.fc2(pred_y1))
        pred_y2 = self.norm4(pred_y2)
        pred_y2 = F.relu(pred_y2)
        pred_y2 = (self.fc3(pred_y2))
        
        return pred_y11,pred_y2


                    
def main(num_epochs, batch_size, lr,FFN, MHA, nnn, sharing,alpha):
    model = NeuralCDE(input_channels=52, hidden_channels=52, output_channels=52, nodes = nnn, Multihead = MHA, feedforward = FFN).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=75, gamma=0.1)

    #train_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(train_data)


    train_dataset = CustomTensorDataset(train_data, (train_label, train_label1))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        model.train()
        for batch_x, batch_y, batch_y1 in train_dataloader:
            

            batch_x, batch_y, batch_y1 = batch_x.to(device), batch_y.to(device), batch_y1.to(device)

            batch_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(batch_x).to(device)
            batch_coeffs = batch_coeffs.to(device)

            aug_data, (batch_y, shuffled_targets_A), (batch_y1, shuffled_targets_B), lam = mixup(batch_coeffs, batch_y, batch_y1, alpha=alpha)
            aug_data = aug_data.to(device)

            optimizer.zero_grad()
            pred_y, pred_y1 = model(aug_data)
            pred_y = pred_y.squeeze(-1).to(device)
            pred_y1 = pred_y1.squeeze(-1).to(device)


            reg_loss = sum(torch.norm(p1 - p2) for p1, p2 in zip(model.cross_attn.parameters(), model.cross_attn1.parameters()))
            
            shuffled_targets_A = shuffled_targets_A.to(device)
            shuffled_targets_B = shuffled_targets_B.to(device)

            loss1 = lam * criterion(pred_y, batch_y) + (1. - lam) * criterion(pred_y, shuffled_targets_A)
            loss2 = lam * criterion(pred_y1, batch_y1) + (1. - lam) * criterion(pred_y1, shuffled_targets_B)


            loss =  loss1 + loss2 + sharing * reg_loss
            loss.backward()
            optimizer.step()

        scheduler.step()
        print(f'Epoch: {epoch} Training loss: {loss.item()}')

        # Evaluation
        model.eval()
        with torch.no_grad():
            test_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(test_data).to(device)
            pred_y, pred_y1 = model(test_coeffs)
            _, predicted_classes = torch.max(pred_y, dim=1)
            task1_accuracy = (predicted_classes == test_label.to(device)).float().mean().item()
            _, predicted_classes1 = torch.max(pred_y1, dim=1)
            task2_accuracy = (predicted_classes1 == test_label1.to(device)).float().mean().item()
            print(f'Task1 Test Accuracy: {task1_accuracy}')
            print(f'Task2 Test Accuracy: {task2_accuracy}')

if __name__ == '__main__':
    main(args.num_epochs, args.batch_size, args.lr, args.FFN, args.MHA, args.nnn, args.sharing, args.alpha)
