# CNN
## 从全连接层到卷积
### 不变性
1. 平移不变性 translation invariance，神经网络的前几层应该对相同的图像区域有相似的反应。
2. 局部性 locality，神经网络的前面几层应该只探索图像中的局部区域，而不过度在意图像中相隔较远区域的关系。
### 多层感知机的限制
认为无论输出还是隐藏表示都拥有空间结构。使用四阶权重。
$$
\begin{align*}
[H]_{i, j} &= [U]_{i, j} + \sum_{k}\sum_{l}[W]_{i, j, k, l}[X]_{k, l} \\
&= [U]_{i, j} + \sum_{a}\sum_{b}[V]_{i, j, a, b}[X]_{i+a, j+b}.
\end{align*}
$$
平移不变性，意味着对象在X里平移仅导致隐藏表示H中的平移。因此跟 i，j 无关。
$$
\begin{align*}
[H]_{i, j} = u + \sum_{a}\sum_{b}[V]_{a, b}[X]_{i+a, j+b}.
\end{align*}
$$
也就是同一个东西应用到不同的（i，j）上。

又由于局部性，只关心 $\Delta$ 范围以内的输入：
$$
[H]_{i, j} = u + \sum_{a=-\Delta}^{\Delta}\sum_{b=-\Delta}^{\Delta}[V]_{a, b}[X]_{i+a, j+b}.
$$
这是一个卷积层。$V$ 被称为卷积核 convolution kernel 或滤波器 filter。

#### 通道
每个输入本质上还有第三个维度，即 rgb 维度。因此需要一组隐藏表示。想象成一些互相堆叠的二维网格 $\rightarrow$ 一系列具有二位张量的通道 channel / 特征映射 feature maps，而各自都向后续层提供一组空间化的学习特征。因此添加第四个坐标。

## 图像卷积
### 互相关运算 cross-correlation
```py
import torch.hub
from torch import nn
from d2l import torch as d2l

def corr2d(X, K):
    """计算二维互相关运算"""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1]-w+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y
        
X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
print(corr2d(X, K))
```
### 卷积层
```py
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
```

### Edge detection
```py
#create 6 * 8 black and white image
X = torch.ones((6, 8))
X[:, 2:6] = 0
print(X)

K = torch.tensor([1.0, -1.0])
Y = corr2d(X, K)
print(Y)

# this can only detect vertical edges, if tranfered:
print(coor2d(X.t(), K))
```

### Learning convolutional kernel
Let's check if we can learn the convolutional kernel only through checking 'input-output' pairs.

Initialize the convolutional kernel randomly. Check square root loss.
```py
# construct a 2d kernel layer. Only one tunnel
conv2d = nn.Conv2d(1, 1, kernal_size=(1, 2), bias=False)

# this 2d convolutional kernel uses a four dimention input and output.(batch size, channels, height, length)
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
lr = 3e-2
for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    # iterate the convolutional kernel
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'epoch {i+1}, loss {l.sum():.3f}')

conv2d.weight.data.reshape((1, 2))
```
#### nn.Conv2d function
`nn.Conv2d` 是 PyTorch 中用于创建二维卷积层的类，其参数如下：

1. `in_channels`：输入图像的通道数。
2. `out_channels`：卷积产生的通道数，也就是输出特征图的数量。
3. `kernel_size`：卷积核的大小，可以是一个整数或者一个元组 `(k_height, k_width)` 来指定卷积核的高度和宽度。
4. `stride`：卷积的步长，可以是一个整数或者一个元组 `(s_height, s_width)` 来指定垂直和水平方向的步长。默认值为1。
5. `padding`：输入图像的填充量，可以是一个整数或者一个元组 `(p_height, p_width)` 来指定垂直和水平方向的填充量。默认值为0。
6. `dilation`：卷积核元素之间的间距，可以是一个整数或者一个元组 `(d_height, d_width)`。默认值为1。
7. `groups`：从输入通道到输出通道的连接数。默认值为1，表示没有分组，所有输入通道都与所有输出通道相连。
8. `bias`：如果为True，则在卷积层中添加一个可学习的偏置参数。默认值为True。
9. `padding_mode`：填充模式，可以是 'zeros', 'reflect', 'replicate' 或 'circular'。默认为 'zeros'。

这些参数共同定义了卷积层的行为，包括如何对输入数据进行卷积操作以及卷积核的配置。通过调整这些参数，可以改变卷积层的学习能力和输出特征图的大小。


## Padding and stride
padding the edge with zero element.
```py
# create a function to pad zeros
def comp_conv2d(conv_2d, X):
    # batchsize=1, channel=1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # omit first two dim - batchsize, channel
    return Y.reshape(Y.shape[2:])

conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
X = torch.rand(size=(8, 8))
print(comp_conv2d(conv2d, X).shape)

# padding=(height, length) -> (row, column)
conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
```
### Stride
the number of the elements slid at a time -> stride.
```py
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=(2, 1), stride=(2, 1))
```

## Multi-inputs and Multi-outputs channel
### multi-inputs
输出后的多 channel 对应数位 (i, j) 相加 -> 1 channel
```py
# one channel K ---apply--> X's channels
def coor2d_multi_in(X, K):
    return sun(d2l.coor2d(x, k) for x, k in zip(X, K))  # zip
```
### multi_outputs
不同通道是对不同特征的响应。对不同的通道给出不同的 kernel.
```py
# K's channels ---apply--> X's channels
def coor2d_multi_in_out(X, K):
    return torch.stack([coor2d_multi_in(X, k) for k in K], 0)

K = torch.stack((K, K+1, K+2), 0)
print(K.shape)
```
#### torch.stack()
参数：

- tensors：一个序列（如列表或元组）的张量，它们需要有相同的形状。

- dim：沿着哪个维度进行堆叠。默认为0，即在最前面添加一个新的维度。

### 1 * 1 convolutional kernel
In this situation, the only calculation happens on the channel dimension.
```py
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h*w))
    K = K.reshape((c_o, c_i))
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, h, w))

# 等价于 corr2d_multi_in_out
```

## 汇聚层 Pooling
最后一层输入全局敏感，逐渐聚合信息。双重目的：
- 降低卷积层对位置的敏感性
- 降低对空间降采样表示的敏感性。

汇聚层也有固定形状，按 stride 移动。但是不包含参数。通常计算汇聚窗口中所有元素的最大值或者平均值（maximum pooling / average pooling）。如此即使平移也不会影响结果

汇聚窗口形状为 p\*q 的汇聚层称为 p\*q 汇聚层。
```py
def pool2d(X, pool_size, mode='max):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode = 'max':
                Y[i, j] = X[i:i + p_h, j : j + p_w].max()
            if mode = 'avg':
                Y[i, j] = X[i:i + p_h, j : j + p_w].mean()    
    return Y

X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
pool2d(X, (2, 2))

pool2d = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(0,1)) 
```

## 卷积神经网络 LeNet
用卷积层代替展开为一维向量：可以保留空间结构，同时模型更简洁，所需的参数更少。

总体来看，LeNet（LeNet-5）由两个部分组成：
- 卷积编码器：由两个卷积层组成；
- 全连接层密集块：由三个全连接层组成。

FRAME: conv 6@28\*28 $\rightarrow$ pooling 6@14\*14 $\rightarrow$ conv 16@10\*10 $\rightarrow$ pooling 16@5\*5 $\rightarrow$ full-connecting \* 3 $\rightarrow$ output

```py
import torch
from torch import nn
from d2l import torch as d2l

net = nn.Sequencial(nn.Conv2d(1, 6, kernal_size=5, padding=2), nn.Sigmoid(),
                     nn.AvgPool2d(kernel_size=2, stride=2),
                     nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
                     nn.AvgPool2d(kernel_size=5, stride=2),
                     nn.Flatten(),      # very important
                     nn.Linear(16*5*5, 120), nn.Sigmoid(),
                     nn.Linear(120, 84), nn.Sigmoid(),
                     nn.Linear(84, 10), nn.Sigmoid())

X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape: \t', X.shape)
```
### LeNet on Fashion-MNIST Dataset
```py
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
```
虽然卷积神经网络的参数较少，但计算成本仍然很高，因为每个参数都参与更多的乘法。通过GPU可以加快训练。

在模型使用GPU计算数据集之前，我们需要将其复制到显存中。
```py
def evaluate_accuracy_gpu(net, data_iter, device=None):
    if isinstance(net, nn.Module):
        net.eval()  # set to evaluating mode
        if not device:
            device = next(iter(net.parameters())).device
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT 微调所需的
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]
```
#### net.eval()
在PyTorch中，`net.eval()` 是一个非常重要的函数，它用于将模型设置为评估模式。当你调用 `net.eval()` 时，模型会禁用一些在训练时使用但在评估时不需要的层，比如 Dropout 和 Batch Normalization。这些层在训练时有助于防止过拟合和加速收敛，但在模型评估或推理时，我们希望使用固定的参数来确保结果的一致性。

以下是 `net.eval()` 的一些关键点：

1. **禁用 Dropout**：在训练时，Dropout 层会随机地关闭一些神经元，以防止模型过拟合。但在评估模式下，我们希望所有的神经元都参与前向传播，因此 Dropout 层会被禁用。这意味着在评估时，所有的神经元都会激活，而不是像训练时那样以一定的概率被丢弃。

2. **改变 Batch Normalization 行为**：Batch Normalization 层在训练时会计算每个 mini-batch 的均值和方差，并使用这些统计数据来规范化层的输入。然而，在评估模式下，由于我们通常对单个样本进行推理，而不是整个 mini-batch，因此会使用训练期间计算的全局均值和方差，而不是单个样本的统计数据。这有助于保持模型在训练和评估时行为的一致性。

3. **不跟踪梯度**：在评估模式下，模型在前向传播中不再跟踪梯度，这可以减少内存消耗，并且不会进行参数更新。

4. **提高推理速度**：由于在评估模式下不进行梯度计算，这可以加速模型的推理过程。

在实际使用中，当你完成模型的训练并准备对其进行评估或进行预测时，你应该确保调用 `net.eval()` 来设置正确的模式。这通常与 `torch.no_grad()` 上下文管理器一起使用，以进一步优化推理过程并减少内存使用。

例如，如果你正在评估一个模型，你应该这样写代码：

```python
model.eval()  # 设置模型为评估模式
with torch.no_grad():  # 关闭梯度计算
    outputs = model(inputs)  # 进行预测或评估
```

总之，`net.eval()` 是 PyTorch 中一个非常有用的函数，它确保了模型在评估和推理时使用正确的行为，从而保证了模型性能的稳定性和准确性。

#### .numel()   （回忆） -> number of elements
在 PyTorch 中，y.numel() 是一个方法，用于返回张量 y 中元素的总数。这个方法非常直接，它会计算张量中所有元素的数量，不考虑张量的维度。

例如，如果你有一个张量 y，它的形状是 (3, 4, 5)，那么 y.numel() 将会返回 3 * 4 * 5 = 60，因为总共有60个元素。

#### Xavier
在PyTorch中，Xavier初始化（也称为Glorot初始化）是一种权重初始化方法，旨在保持激活函数的方差在网络的前向传播和反向传播过程中大致相同，以避免梯度消失或梯度爆炸的问题。这种方法是由Xavier Glorot和Yoshua Bengio在2010年提出的。

Xavier初始化的数学原理基于这样一个观察：如果输入和权重都是零均值的高斯分布，那么线性组合的方差将是输入方差和权重方差的乘积。为了保持每一层的输出方差接近于其输入方差，Xavier初始化建议设置权重的初始方差为：

\[ \text{Var}(w) = \frac{2}{n_{\text{in}} + n_{\text{out}}} \]

其中 \( n_{\text{in}} \) 是输入单元的数量，\( n_{\text{out}} \) 是输出单元的数量。这样，无论 \( n_{\text{in}} \) 和 \( n_{\text{out}} \) 的大小如何，这一层的输出方差都接近于其输入方差。

在PyTorch中，可以使用`torch.nn.init`模块中的`xavier_uniform_`或`xavier_normal_`函数来应用Xavier初始化。`xavier_uniform_`使用均匀分布，而`xavier_normal_`使用正态分布。以下是如何在PyTorch中使用Xavier初始化的示例：

```python
import torch
import torch.nn as nn

# 假设我们有一个简单的全连接层
fc_layer = nn.Linear(in_features=256, out_features=512)

# 应用Xavier均匀初始化
nn.init.xavier_uniform_(fc_layer.weight)

# 或者应用Xavier正态初始化
# nn.init.xavier_normal_(fc_layer.weight)
```

在这段代码中，`fc_layer`是一个全连接层，它有256个输入特征和512个输出特征。通过调用`xavier_uniform_`或`xavier_normal_`，我们可以将Xavier初始化应用到该层的权重上。这种初始化方法特别适用于激活函数是线性的情况，比如tanh或sigmoid。然而，对于ReLU激活函数，Xavier初始化可能不是最佳选择，因为它的推导是基于激活函数是线性的假设，而ReLU是非线性的。

我们调用Xavier进行随机初始化。
```py 
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """use GPU to train models"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], legend=['tarin loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()           # 直接把存在“参数.grad”里的所有梯度清零了，
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():           # 暂时禁用梯度计算
                metric.add(l*X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (train_l, train_acc, None))
                test_acc = evaluate_accuracy_gpu(net, test_iter)
                animator.add(epoch+1, (None, None, test_acc))
            print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, teat acc {test_acc:.3f}')
            print(f'{metric[2]*num_epochs/timer.sum():.1f} examples/sec on {str(device)}')

lr, num_epochs = 0.9, 10
train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```







