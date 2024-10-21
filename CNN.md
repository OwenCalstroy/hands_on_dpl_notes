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

# this 2d convolutional kernel uses a four dimention input and output.(batch size, tunnels, height, length)
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
lr = 3e-2
for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    #iterate the convolutional kernel
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
