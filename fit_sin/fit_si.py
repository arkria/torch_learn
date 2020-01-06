import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(1, 50)
        # self.layer2 = nn.Linear(50, 50)
        self.layer3 = nn.Linear(50, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        # x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x


x = torch.unsqueeze(torch.linspace(0, 6.29, 150), dim=1)
y = x.sin()+0.3*torch.rand(x.size())
plt.scatter(x.data.numpy(), y.data.numpy())
plt.show()

batch_size = 29

torch_dataset = torch.utils.data.TensorDataset(x, y)
train_db, val_db = torch.utils.data.random_split(torch_dataset, [145, 5])
train_loader = torch.utils.data.DataLoader(
dataset=train_db, # torch TensorDataset format
batch_size=batch_size, # mini batch size
shuffle=True, # random shuffle for training
num_workers=2, # subprocesses for loading data
)
val_loader = torch.utils.data.DataLoader(
dataset=val_db, # torch TensorDataset format
batch_size=batch_size, # mini batch size
shuffle=True, # random shuffle for training
num_workers=2, # subprocesses for loading data
)


plt.ion()
net = Net()
optimizer = optim.Adam(net.parameters(), lr=0.1)
criteon = nn.MSELoss()

for epoch in range(100):
    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.view(x.size(0), 1)
        print("size:", len(x))
        out = net(x)
        loss = criteon(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % 1 == 0:
            print("epoch: {}, batch idx: {}, loss: {}".format(
            epoch, batch_idx, loss.item()))
            plt.cla()
            plt.scatter(x.data.numpy(), y.data.numpy())
            plt.plot(x.data.numpy(), out.data.numpy(), 'ro', lw=5)
            plt.text(0.5, 0, 'Loss=%.4f' % loss.item(), fontdict={'size': 20, 'color': 'red'})
            plt.pause(0.1)