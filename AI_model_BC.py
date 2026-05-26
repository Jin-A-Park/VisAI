import numpy as np
import pandas as pd
import torch
import torch.nn as  nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from copy import deepcopy

import visAI

# Connect to the visAI
visAI.connect('127.0.0.1')
visAI.get_log('Start')

# Load and preprocess the dataset
cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['class'] = cancer.target

data = torch.from_numpy(df.values).float()
x = data[:, :-1]
y = data[:, -1:]

# Split the dataset into train, valid, test sets
ratios = [0.6, 0.2, 0.2]
train_cnt = int(data.size(0) * ratios[0])
valid_cnt = int(data.size(0) * ratios[1])
test_cnt = data.size(0) - train_cnt - valid_cnt
cnts = [train_cnt, valid_cnt, test_cnt]

indices = torch.randperm(data.size(0))
x = torch.index_select(x, dim=0, index=indices)
y = torch.index_select(y, dim=0, index=indices)

x = x.split(cnts, dim=0)
y = y.split(cnts, dim=0)

# Standardize features
scaler = StandardScaler()
scaler.fit(x[0].numpy())
x = [torch.from_numpy(scaler.transform(xi.numpy())).float() for xi in x]

# Define the model
class MyModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 25)
        self.leaky_relu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(25, 20)
        self.leaky_relu2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(20, 15)
        self.leaky_relu3 = nn.LeakyReLU()
        self.fc4 = nn.Linear(15, 10)
        self.leaky_relu4 = nn.LeakyReLU()
        self.fc5 = nn.Linear(10, 5)
        self.leaky_relu5 = nn.LeakyReLU()
        self.fc6 = nn.Linear(5, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.leaky_relu1(x)
        x = self.fc2(x)
        x = self.leaky_relu2(x)
        x = self.fc3(x)
        x = self.leaky_relu3(x)
        x = self.fc4(x)
        x = self.leaky_relu4(x)
        x = self.fc5(x)
        x = self.leaky_relu5(x)
        x = self.fc6(x)
        x = self.sigmoid(x)
        return x

    def get_activation_output(self, x):
        x = self.fc1(x)
        x = self.leaky_relu1(x)
        x = self.fc2(x)
        return self.leaky_relu2(x)

# Initialize model, optimizer, and hyperparameters
model = MyModel(x[0].size(-1), y[0].size(-1))
optimizer = optim.Adam(model.parameters())
n_epochs = 100
batch_size = 32
lowest_loss = np.inf
early_stop = 10
lowest_epoch = np.inf
best_model = None

# Training loop
for epoch in range(n_epochs):
    indices = torch.randperm(x[0].size(0))
    x_ = torch.index_select(x[0], dim=0, index=indices)
    y_ = torch.index_select(y[0], dim=0, index=indices)

    x_ = x_.split(batch_size, dim=0)
    y_ = y_.split(batch_size, dim=0)

    train_loss = 0
    valid_loss = 0
    all_train_y_hat, all_train_y = [], []

    # Training step
    for x_batch, y_batch in zip(x_, y_):
        y_hat = model(x_batch)
        loss = F.binary_cross_entropy(y_hat, y_batch)
        train_loss += float(loss)

        # Track predictions and targets
        all_train_y_hat.extend(y_hat.detach().cpu().numpy())
        all_train_y.extend(y_batch.detach().cpu().numpy())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(x_)

    # Validation step
    with torch.no_grad():
        valid_batches = x[1].split(batch_size, dim=0)
        valid_targets = y[1].split(batch_size, dim=0)

        all_valid_y_hat, all_valid_y = [], []
        for x_batch, y_batch in zip(valid_batches, valid_targets):
            y_hat = model(x_batch)
            loss = F.binary_cross_entropy(y_hat, y_batch)
            valid_loss += float(loss)

            all_valid_y_hat.extend(y_hat.detach().cpu().numpy())
            all_valid_y.extend(y_batch.detach().cpu().numpy())

        valid_loss /= len(valid_batches)

    # Update visAI
    visAI.get_loss(n_epochs, epoch, train_loss, valid_loss)
    visAI.get_acc(n_epochs, epoch, all_train_y_hat, all_train_y, all_valid_y_hat, all_valid_y)
    visAI.get_weight(n_epochs, epoch, model.fc3.weight)

    activation_output = model.get_activation_output(x[1][0])
    visAI.get_act(n_epochs, epoch, activation_output)

    grad_norm = model.fc3.weight.grad.norm(2).item() if model.fc3.weight.grad is not None else 0
    visAI.get_gradnorm(n_epochs, epoch, grad_norm)

    # Early stopping
    if valid_loss < lowest_loss:
        lowest_loss = valid_loss
        lowest_epoch = epoch
        best_model = deepcopy(model.state_dict())
    elif epoch - lowest_epoch > early_stop:
        break

visAI.get_log('END')
visAI.disconnect()