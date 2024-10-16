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
```

## MLP easily done
```py
import torch
from torch import nn
from d2l import torch as d2l

net == nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:                    # m is the elements from Sequential
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);

batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=lr)

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

## 模型选择、欠拟合和过拟合
将模型在训练数据上拟合的比潜在分布中更接近的现象称为过拟合 (overfilting)，用于对抗过拟合的技术称为正则化 (regularization)。

训练误差 (training error)：模型在训练数据集上得到的误差

泛化误差 (generalization error)：模型应用在同样从原始样本分布中抽取的无限多数据样本时误差

独立同分布假设 (i.i.d assumption) ：假设训练数据和测试数据都是从相同的分布中独立提取的，意味着对于数据进行采样的过程没有进行“记忆”。

### 模型选择
为了确定模型中的最佳模型，我们通常会使用验证集。 -> validation dataset / validation set

K折交叉验证：原始训练数据分成K个不重叠的子集，然后执行K次模型训练和验证（K - 1 个子集上训练，在剩下一个子集上验证），取平均来估计误差。

### 多项式回归
```py
import math
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l
```
#### generate dataset
Given x, we are going to use the following sentence to generate the training and testing data's labels:
$$ y = 5 + 1.2x - 3.4\frac{x^2}{2!} + 5.6\frac{x^3}{3!} + \epsilon$$
where $\epsilon$ ~ $N(0, 0.1)$
```py
max_degree = 20 # 多项式的最大阶数
n_train, n_test = 100, 100 # 训练和测试数据集大小
true_w = np.zeros(max_degree) # 分配大量空间
ture_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

features = np.random.normal(size=(n_train + n_test, 1))
np.random.shuffle(features)
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i + 1)    # gamma(n) = (n-1)!
# labels's dimention: (n_train+n_test,)
labels = np.dot(poly_features, true_w)
labels += np.random.normal(scale=0.1, size=labels.shape)

# Numpy ndarray -> tensor
true_w, features, poly_features, labels = [torch.tensor(x, dtype=torch.float32) for x in [true_w, features, poly_features, labels]]
print(features[:2], poly_features[:2, :], labels[:2])
```
#### Train and test the model
```py
def evaluate_loss(net, data_iter, loss):
    """evaluate the loss from te given dataset model"""
    metric = d2l.Accumulator(2) # 分别为损失的总和，样本数量
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)\
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    reuturn metriv[0] / metric[1]

# def training function
def train(train_features, test_features, train_labels, test_labels, num_epochs=400):
    loss = nn.MSELoss(reduction='none')
    input_shape = train_features.shape[-1]
    # 不设置偏置，因为我们已经在多项式中实现了它
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labes.reshape(-1, 1)), batch_size)
    test_iter = d2l.load_array((test_features, test_labes.reshape(-1, 1)), batch_size, is_train=False)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log', xlim=['train', 'test'])
    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch+1, (evaluate_loss(not, train_iter, loss), evaluate_loss(net, test_iter, loss)))
    print('weight:', net[0].weight.data.numpy())

# 三阶多项式函数拟合
# 从多项式特征中选择前四个维度
train(poly_features[:n_train, :4], poly_features[n_train:, :4], labels[:n_train], labels[n_train:])

# 线性函数拟合（欠拟合）
# 选择前两个维度：1，x
train(poly_features[:n_train, :2], poly_features[n_train:, :2], labels[:n_train], labels[n_train:])

# 高阶多项式函数拟合（过拟合）
# 选取所有维度
train(poly_features[:n_train, :], poly_features[n_train:, :], labels[:n_train], labels[n_train:], num_epochs=1500)
```

## 权重衰减
在机器学习和深度学习中，保证权重（weights）的“最小”并不是一个普遍追求的目标。相反，权重的选择通常是为了在给定的数据集上优化模型的性能，比如提高准确率、召回率或其他性能指标。然而，有一些情况下，我们会考虑对权重进行正则化或约束，以防止某些不希望出现的问题，以下是一些可能的原因：

1. **防止过拟合**：过拟合发生在模型在训练数据上表现很好，但在未见过的测试数据上表现差的情况下。通过限制权重的大小（例如，使用L1或L2正则化），可以减少模型的复杂度，使其更加泛化。

2. **提高数值稳定性**：过大的权重可能导致数值计算中的溢出或不稳定，尤其是在深度网络中。通过限制权重的大小，可以提高模型的数值稳定性。

3. **加快收敛速度**：在某些情况下，较小的权重可以帮助优化算法（如梯度下降）更快地收敛，因为权重更新的步长会更小，从而减少训练过程中的振荡。

4. **促进特征选择**：在某些模型（如带有L1正则化的线性模型）中，权重的稀疏性可以帮助进行特征选择，即自动将不重要的特征的权重设置为零。

5. **避免梯度消失或爆炸**：在深度学习中，如果权重过大，可能会导致梯度消失或爆炸，这会阻碍有效的反向传播。通过限制权重的大小，可以减轻这个问题。

6. **实现模型的可解释性**：较小的权重通常与模型参数的可解释性相关，因为它们可以减少模型对单个特征的依赖。

7. **满足特定业务需求**：在某些业务场景中，可能需要模型的权重保持在一定的范围内，以满足特定的业务逻辑或合规性要求。

总的来说，“保证权重最小”并不是机器学习中的一个普遍目标，而是在特定情况下，通过正则化或约束权重来实现更好的模型性能和稳定性。在实践中，权重的选择应该基于模型的性能指标和特定的应用需求。

要保证权重向量最小，最常用的方法是将其范数作为惩罚项加到最小化损失的问题中。
$$L(\vec w,b) = \frac{1}{n}\sum_{i=1}^{n}0.5*(\vec w^T\vec x^{(i)}+b-y^{(i)})^2$$
$$=>\ \ L(\vec w, b)+\frac{\lambda}{2}||\vec w||^2$$
$L2$ 正则化回归SGD：
$$\vec w \leftarrow (1-\eta \lambda)\vec w - \frac{\eta}{|B|}\sum_{i\in B}\vec x^{(i)}(\vec w^T\vec x^{(i)}+b-y^{(i)})$$
实现
```py
import matplotlib.pyplot as plt
import torch
from torch import nn
from d2l import torch as d2l 
```
generate data from:
$$y=0.05+\sum_{i=1}^d 0.01x_i+\epsilon$$
$\epsilon$ is the noise.
```py
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, True_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)
```
Realization
```py
def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]

def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2

def train(lambd):
    w, b = init_params()
    net, loss = lambda x: d2l.linreg(X, w, b), d2l.squared_loss
    num_eqochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log', xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch+1, (d2l.evaluate_loss(net, train_iter, loss), d2l.evaluate_loss(net, test_iter, loss)))
    print("w 的 L2 范数是：", torch.norm(w).item())

# lambd = 0 -> ignore the regularization.
```

### Easy access
```py
def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.paraameters():
        param.data.normal_()
    loss = nn.MSELoss(reduction='none') #reduction='none' 保留每个样本的损失值，而不是将它们自动求和或取平均。
    num_epochs, lr = 100, 0.003
    # bias do not decrease
    trainer = torch.optim.SGD([{"params":net[0].weight,'weight_decay':wd}, {"params":net[0].bias}, lr=lr])
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log', xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.mean().backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch+1, (d2l.evaluate_loss(net, train_iter, loss), d2l.evaluate_loss(net, test_iter, loss)))
    print("w 的 L2 范数是：", torch.norm(w).item())
```
注意！在这里似乎是对于每一组 (X, y) 都直接进行权重的优化。而前一种则是将所有loss加起来后统一求loss。

本例是在线学习 (online)

在深度学习中，通常有两种方式来计算和更新模型的权重：

1. **累积（Batch）模式**：在每个epoch中，模型的权重会根据整个训练集（或一个大批次）上的累积梯度进行更新。这种方法通常称为批量梯度下降（Batch Gradient Descent）。

2. **逐样本（Online）模式**：模型的权重在处理完每个样本或小批次后立即更新。这种方法通常称为随机梯度下降（Stochastic Gradient Descent，SGD）。

### 结合两者的优点：

在实际应用中，通常会使用小批次（Mini-batch）SGD，这是一种折中的方法。它结合了批量梯度下降的稳定性和SGD的快速更新。在小批次SGD中，权重的更新是基于每个小批次的累积梯度，而不是整个训练集或单个样本。

在您的代码中，如果 `train_iter` 是一个迭代器，它每次返回一个小批次的数据，那么 `trainer.step()` 实际上是在处理完每个小批次后更新权重，而不是在处理完所有样本后。这是深度学习中常见的做法，因为它平衡了计算效率和模型更新的速度。

## 暂退法 Dropout
当面对更多的特征而样本不足，往往会过拟合；当给出更多样本而不是特征，通常不会过拟合。

泛化性和灵活性之间的这种基本权衡被描述为偏差-方差权衡 bias-variance tradeoff。线性模型有很高的偏差：只能表示一小类函数，而模型方差很低：在不同随机数据上可得出相似的结果

暂退法：在训练过程中，在计算后续层之前向网络的每一层注入噪声。当训练一个有多层的深层网络时，注入噪声只会在输入-输出映射上增加平滑性。
在训练过程中dropout 部分神经元。

$p$ 的概率 $h' = 0$，其它情况 $h'=\frac{h}{1-p}$. 期望 $E[h'] = h$.

是概率而非比例。

### Realization
```py
import torch
from torch import nn
from d2l import torch as d2l

# drop out at the possibility of 'dropout'
def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1:
        return torch.zeros_like(X)
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape) > dropout).float()  # this idea is extraordinary.
    return mask * X / (1.0 - dropout)

X = torch.arange(16, dtype = torch.float32).reshape((2, 8))
print(X)
print(dropout_layer(X, 0.))
print(dropout_layer(X, 0.5))
print(dropout_layer(X, 1.)

num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

dropout1, dropout2 = 0.2, 0.5

class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2, is_training=True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()
    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        # Only dropout in training
        if self.training == True:
            # add dropout after first layer
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training == True:
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out
    
net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)

num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss(reduction='none')
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

### Easily done
```py
net = nn.Sequential(nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        # 在第一个全连接层之后添加一个dropout层 nn.Dropout(dropout1),
        nn.Linear(256, 256),
        nn.ReLU(),
        # 在第二个全连接层之后添加一个dropout层 nn.Dropout(dropout2),
        nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);
```
#### net.apply()

在PyTorch中，`net.apply()` 方法是一个非常方便的工具，它允许你对神经网络中的所有模块（包括子模块）递归地执行一个指定的函数。这通常用于初始化网络中的权重和偏置，或者对它们进行某种形式的变换。

`net.apply()` 方法的基本语法如下：
```python
net.apply(fn)
```
- `net`：你的神经网络模型，它应该是 `torch.nn.Module` 的一个实例。
- `fn`：你想要应用到每个模块的函数。这个函数应该接受一个 `torch.nn.Module` 作为参数。

训练和测试

```py
trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```