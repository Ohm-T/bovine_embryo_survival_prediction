import torch
import random
import numpy as np
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.nn.functional as F
from torchvision.models import resnet18, resnet101
from sklearn.preprocessing import OneHotEncoder

class VideoDataset(Dataset):
    def __init__(self, videos_train, y_true, transform=None, pred_times = [27, 32, 37, 40, 44, 48, 53, 58, 63, 94]):      
        self.transform = transform
        self.videos_train = videos_train
        self.labels = y_true
        self.pred_times = pred_times
        
    def __len__(self):
        return len(self.video_train)
    def __getitem__(self, idx):
        frames = []
        lbl = self.labels[idx]
        video = self.videos_train[idx]
        for time in self.pred_times:
            frame = video.read_frame(time)
            frames.append(frame)
        
        if self.transform is not None:
            seed = np.random.randint(1e9)        
            frames_tr = []
            for frame in frames:
                random.seed(seed)
                np.random.seed(seed)
                frame = self.transform(frame)
                frames_tr.append(frame)

        frames = torch.stack(frames)
        #frames_tr = torch.stack(frames_tr)
        return frames, lbl

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x  

# Method 1 : CNN + LSTM
class CNN_LSTM(nn.Module):
    def __init__(self, nhid=256, n_layers=2, n_class=8, dropout=0.2):
        """
        :param nhid: Dimension of hidden layer
        :param n_layers: Number of LSTM layers
        :param n_class: Number of classes
        :param dropout: Dropout rate
        """
        super(CNN_LSTM, self).__init__()
        self.resnet = resnet18(pretrained=True)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = Identity()
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=nhid, num_layers=n_layers)
        self.fc1 = nn.Linear(nhid, nhid//2)
        self.fc2 = nn.Linear(nhid//2, n_class)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_in):
        hid = None
        bs, n_frames, c, h, w = x_in.size()
        x = self.resnet(x_in[:, 0])
        out, (hn, cn) = self.lstm(x.unsqueeze(1), hid)  
        for i in range(1, n_frames):
            x = self.resnet(x_in[:, i])
            out, (hn, cn) = self.lstm(x.unsqueeze(1), (hn, cn))
        out = self.dropout(out)
        out = F.relu(self.fc1(out[:, -1]))
        out = self.fc2(out)
        return out

    def fit(self, videos_train, labels_train, batch_size=64, epochs=50, lr=0.001, device = 'cpu', verbose=True):
        """
        Train the model with simple fit method as sklearn or keras.
        :param x_train: Training data.
        :param y_train: Training labels.
        :param batch_size: Batch size.
        :param epochs: Number of epochs.
        :param lr: Learning rate.
        """
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.device = torch.device(device)

        # Loading data with ohe of labels
        labels_ohe_train = labels_train.reshape(-1, 1)
        enc = OneHotEncoder()
        enc.fit(labels_ohe_train)
        labels_ohe_train = enc.transform(labels_ohe_train)
        train_data = DataLoader(videos_train, labels_ohe_train, batch_size=batch_size)

        self.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_idx, (x, y) in enumerate(train_data):
                x, y = x.to(device), y.to(device)
                self.optimizer.zero_grad()
                output = self.forward(x)
                loss = self.loss_fn(output, y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                print(output)
                
                if verbose and batch_idx % 5 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(x), len(x), correct, loss.item() * 100))
                        


    def predict(self, videos_test, labels_test, batch_size=64, device = 'cpu'):
        """
        Predict the labels of the test data.
        :param x_test: Test data.
        :param device: Device to use.
        :return: Predicted labels.
        """
        labels_ohe_test = labels_test.reshape(-1, 1)
        enc = OneHotEncoder()
        enc.fit(labels_ohe_test)
        labels_ohe_test = enc.transform(labels_ohe_test)

        with torch.no_grad():
            test_data = DataLoader(videos_test, labels_ohe_test, batch_size=batch_size)
            self.eval()
            y_pred = []
            for i, (x, y) in enumerate(test_data):
                out = self(x)
                
        