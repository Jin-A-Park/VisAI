import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import visAI
visAI.connect('127.0.0.1')
visAI.get_log('Start')

train = datasets.MNIST(
  '../data', train=True, download=True,
  transform=transforms.Compose([
    transforms.ToTensor(),
  ]),
)
test = datasets.MNIST(
  '../data', train=False,
  transform=transforms.Compose([
    transforms.ToTensor(),
  ]),
)

def plot(x):
  img = (np.array(x.detach().cpu(), dtype='float')).reshape(28,28)

  plt.imshow(img, cmap='gray')
  plt.show()
    
#plot(train.data[0])

x = train.data.float() / 255.
y = train.targets

x = x.view(x.size(0), -1)
#print(x.shape, y.shape)

input_size = x.size(-1)
output_size = int(max(y)) + 1

#print('input_size: %d, output_size: %d' % (input_size, output_size))
# Train / Valid ratio
ratios = [.8, .2]

train_cnt = int(x.size(0) * ratios[0])
valid_cnt = int(x.size(0) * ratios[1])
test_cnt = len(test.data)
cnts = [train_cnt, valid_cnt]

#print("Train %d / Valid %d / Test %d samples." % (train_cnt, valid_cnt, test_cnt))

indices = torch.randperm(x.size(0))

x = torch.index_select(x, dim=0, index=indices)
y = torch.index_select(y, dim=0, index=indices)

x = list(x.split(cnts, dim=0))
y = list(y.split(cnts, dim=0))

x += [(test.data.float() / 255.).view(test_cnt, -1)]
y += [test.targets]
#for x_i, y_i in zip(x, y):
  #print(x_i.size(), y_i.size())

#---------------------------------------model------------------------------------------
'''
model = nn.Sequential(
  nn.Linear(input_size, 500),#500neurons
  nn.LeakyReLU(),
  nn.Linear(500, 400),#400neurons
  nn.LeakyReLU(),
  nn.Linear(400, 300),#300neurons
  nn.LeakyReLU(),
  nn.Linear(300, 200),#200neurons
  nn.LeakyReLU(),
  nn.Linear(200, 100),
  nn.LeakyReLU(),
  nn.Linear(100, 50),
  nn.LeakyReLU(),
  nn.Linear(50, output_size),
  nn.LogSoftmax(dim=-1),
)
'''
# Custom Model Class
class MyModel(nn.Module):
  def __init__(self, input_size, output_size):
    super(MyModel, self).__init__()
    self.fc1 = nn.Linear(input_size, 500)
    self.leaky_relu1 = nn.LeakyReLU()
    self.fc2 = nn.Linear(500, 400)
    self.leaky_relu2 = nn.LeakyReLU()
    self.fc3 = nn.Linear(400, 300)
    self.leaky_relu3 = nn.LeakyReLU()
    self.fc4 = nn.Linear(300, 200)
    self.leaky_relu4 = nn.LeakyReLU()
    self.fc5 = nn.Linear(200, 100)
    self.leaky_relu5 = nn.LeakyReLU()
    self.fc6 = nn.Linear(100, 50) 
    self.leaky_relu6 = nn.LeakyReLU()
    self.fc7 = nn.Linear(50, output_size)
    self.log_softmax = nn.LogSoftmax(dim=-1)
    
    self.output = 0

  def forward(self, x):
    x = self.fc1(x)
    x = self.leaky_relu1(x)
    x = self.fc2(x)
    x = self.leaky_relu2(x)
    output = x
    x = self.fc3(x)
    x = self.leaky_relu3(x)
    x = self.fc4(x)
    x = self.leaky_relu4(x)
    x = self.fc5(x)
    x = self.leaky_relu5(x)
    x = self.fc6(x)
    x = self.leaky_relu6(x)
    x = self.fc7(x)
    x = self.log_softmax(x)
    return x
  
  def get_output(self, x):
    x = self.fc1(x)
    x = self.leaky_relu1(x)
    x = self.fc2(x)
    leaky_relu2_output = self.leaky_relu2(x)
    return leaky_relu2_output

#---------------------------------------Initialize------------------------------------------
model = MyModel(input_size, output_size)
crit = nn.NLLLoss()
optimizer = optim.Adam(model.parameters())
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
model = model.to(device)
x = [x_i.to(device) for x_i in x]
y = [y_i.to(device) for y_i in y]

#---------------------------------------Hyperparameter------------------------------------------
n_epochs = 100
batch_size = 256
print_interval = 10

from copy import deepcopy
lowest_loss = np.inf
best_model = None
early_stop = 50
lowest_epoch = np.inf


#---------------------------------------Training loop------------------------------------------
train_history, valid_history = [], []
all_train_y_hat = []
all_train_y = []
all_valid_y_hat = []
all_valid_y = []

for i in range(n_epochs):
  indices = torch.randperm(x[0].size(0)).to(device)
  x_ = torch.index_select(x[0], dim=0, index=indices)
  y_ = torch.index_select(y[0], dim=0, index=indices)
  
  x_ = x_.split(batch_size, dim=0)
  y_ = y_.split(batch_size, dim=0)
  
  train_loss, valid_loss = 0, 0
  y_hat = []
  
  all_train_y_hat = []
  all_train_y = []
  
  for x_i, y_i in zip(x_, y_):
    y_hat_i = model(x_i)
    #print(leaky_relu2_output)
    loss = crit(y_hat_i, y_i.squeeze())
    
    all_train_y_hat.extend(y_hat_i.detach().cpu().numpy())  # Detach and move to CPU
    all_train_y.extend(y_i.detach().cpu().numpy()) 
    
    optimizer.zero_grad()
    loss.backward()  
    optimizer.step()        
    train_loss += float(loss) # This is very important to prevent memory leak.

  train_loss = train_loss / len(x_)
  #visAI.get_loss(i, train_loss)

  with torch.no_grad():
    x_ = x[1].split(batch_size, dim=0)
    y_ = y[1].split(batch_size, dim=0)
    
    valid_loss = 0
    
    for x_i, y_i in zip(x_, y_):
      y_hat_i = model(x_i)
      loss = crit(y_hat_i, y_i.squeeze())
      all_valid_y_hat.extend(y_hat_i.detach().cpu().numpy())  # Detach and move to CPU
      all_valid_y.extend(y_i.detach().cpu().numpy()) 
      valid_loss += float(loss)
      
      y_hat += [y_hat_i]       
  valid_loss = valid_loss / len(x_)
  
  train_history += [train_loss]
  valid_history += [valid_loss]
  '''
  print(i+1)
  if (i + 1) % print_interval == 0:
    print('Epoch %d: train loss=%.4e  valid_loss=%.4e  lowest_loss=%.4e' % (
      i + 1,
      train_loss,
      valid_loss,
      lowest_loss,
    )) 
  ''' 
  '''
  if valid_loss <= lowest_loss:
    lowest_loss = valid_loss
    lowest_epoch = i
    
    best_model = deepcopy(model.state_dict())
  else:
    if early_stop > 0 and lowest_epoch + early_stop < i + 1:
      print("There is no improvement during last %d epochs." % early_stop)
      break
  '''
  
  visAI.get_loss(n_epochs, i, train_loss, valid_loss)
  visAI.get_acc(n_epochs, i, all_train_y_hat, all_train_y, all_valid_y_hat, all_valid_y)
  visAI.get_weight(n_epochs, i, model.fc3.weight)
  output = model.get_output(x_[0][0])
  visAI.get_act(n_epochs, i, output)
  grad_norm = model.fc3.weight.grad.data.norm(2).item()
  visAI.get_gradnorm(n_epochs, i, grad_norm)
visAI.get_log('END')
visAI.disconnect()