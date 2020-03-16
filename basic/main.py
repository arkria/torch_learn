import torch
import torch.nn as nn
import numpy as np

# 1. auto_grad
w = torch.tensor(3., requires_grad=True)
x = torch.tensor(1., requires_grad=True)
b = torch.tensor(2., requires_grad=True)

y = w * x + b
y.backward()
print('dy/dx', x.grad)
print('dy/dw', w.grad)
print('dy/db', b.grad)

# 2. auto_grad
x = torch.randn(10, 3)  # b, f
y = torch.randn(10, 2)  # b, l

linear = nn.Linear(3, 2)
print('iter 0 weight ', linear.weight)
print('iter 0 bias ', linear.bias)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(linear.parameters(), lr=0.001)

pred = linear(x)
loss = criterion(pred, y)
loss.backward()
print('grad dL/dw', linear.weight.grad)
print('grad dL/db', linear.bias.grad)
print('loss', loss.item())

optimizer.step()
pred = linear(x)
loss = criterion(pred, y)
print('iter 1 weight ', linear.weight)
print('iter 1 bias ', linear.bias)
print('grad dL/dw', linear.weight.grad)
print('grad dL/db', linear.bias.grad)
print('loss', loss.item())
