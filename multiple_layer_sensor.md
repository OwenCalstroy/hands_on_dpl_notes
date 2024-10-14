# Multilayer Preceptron
## MLP
### activation function
通过加权并加上偏置确定神经元是否应激活
```py
import matplotlib.pyplot as plt
import torch
from d2l import torch as d2l
```
#### ReLU
```py
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)
d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))

y.backward(torch.ones_like(x), retain_graph=True) # ones_like 矩阵权重
d2l.plot(x.detach() ,x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
```
#### Sigmoid
```py
y = torch.sigmoid(x)
d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
x.grad.data.zero_()
y.backward(torch.ones_like(x), retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
```
#### tanh

$$\text{tanh}(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = \frac{1 - exp(-2x)}{1 + exp(-2x)}$$

```py
y = torch.tanh(x)
d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))
x.grad.data.zero_()
y.backward(torch.ones_like(x), retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
```

## MLP realization
```py
import toech
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

### Initialize model parameters
```py
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hidens, requires_grad=True))
W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]
```
### Relu
```py
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)      # neat!
```
### model
```py
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X@W1 + b1)     # @ 是矩阵乘法
    return (H@W2 + b2)
```

### loss function
```py
loss = nn.CrossEntrophyLoss(reduction='none')
```
### Training
```py
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)



