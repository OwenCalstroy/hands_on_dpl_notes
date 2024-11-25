# Modern Convolutional Neural Network
## 深度卷积神经网络 Alexnet
### 支持向量机 support vector machines
支持向量机（Support Vector Machine，简称SVM）是一种监督学习算法，主要用于分类问题，但也可以用于回归问题（称为支持向量回归，Support Vector Regression，SVR）。SVM是由Vapnik和Chervonenkis在1963年首次提出的，它是基于统计学习理论中的结构风险最小化原则来构建的。

SVM的核心思想是找到一个超平面（在二维空间中是一条直线，在三维空间中是一个平面，在更高维空间中是一个超平面），这个超平面能够将不同类别的数据点尽可能地分隔开，并且具有最大的间隔（即距离最近的点，这些点被称为支持向量）。以下是SVM的一些关键概念：

1. **超平面（Hyperplane）**：在n维空间中，超平面是n-1维的。对于二维空间，它是一条直线；对于三维空间，它是一个平面。

2. **间隔（Margin）**：间隔是指数据点到分隔超平面的最短距离。SVM的目标是最大化这个间隔。

    在SVM中，间隔定义为数据点到分隔超平面的最短距离。这个距离是衡量分类器泛化能力的重要指标，因为一个较大的间隔意味着模型对于未见过的数据具有更好的预测能力。间隔的大小由最近的那些数据点（支持向量）决定，它们位于超平面的两侧，并且与超平面的距离最短。

    对于线性可分的情况，SVM的目标是找到一个超平面，使得不同类别的数据点被完全分开，并且这个间隔最大化。数学上，这可以表示为：

    $$ \max_{\mathbf{w}, b} \frac{2}{\|\mathbf{w}\|}$$

    受到以下约束：

    $$ y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1, \quad \forall i $$

    其中，$\mathbf{w}$ 是超平面的法向量，$b$ 是偏置项，$\|\mathbf{w}\|$ 是$\mathbf{w}$的欧几里得范数，$y_i$ 是第$i$个数据点的类别标签，$\mathbf{x}_i$ 是第$i$个数据点的特征向量。

3. **支持向量（Support Vectors）**：这些是距离超平面最近的点，它们决定了超平面的位置和方向。

4. **核函数（Kernel Function）**：SVM可以处理线性不可分的数据，通过使用核函数将数据映射到更高维的空间中，使其线性可分。

    核函数是SVM处理非线性问题的关键。在许多实际问题中，数据并不是线性可分的，这时可以通过核函数将原始数据映射到一个更高维的特征空间中，在这个新的空间里，数据可能是线性可分的。常用的核函数包括：

    - **线性核**：$ K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i \cdot \mathbf{x}_j $
    - **多项式核**：$ K(\mathbf{x}_i, \mathbf{x}_j) = (\gamma \mathbf{x}_i \cdot \mathbf{x}_j + r)^d $，其中$\gamma$是缩放因子，$r$是偏置项，$d$是多项式的度数。
    - **径向基函数（RBF）核**：$ K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2) $，其中$\gamma$是一个参数，控制了函数的宽度。
    - **Sigmoid核**：$ K(\mathbf{x}_i, \mathbf{x}_j) = \tanh(\gamma \mathbf{x}_i \cdot \mathbf{x}_j + r) $

    核函数的选择对SVM的性能有重要影响，不同的核函数适用于不同类型的数据和问题。

5. **软间隔（Soft Margin）**：在实际应用中，数据可能不是完全线性可分的。SVM通过引入松弛变量（slack variables）允许一些数据点违反间隔规则，这称为软间隔。

    在现实世界中，数据往往是非线性可分的，即不存在一个超平面可以完美地将所有数据点分开。为了处理这种情况，SVM引入了软间隔的概念，允许一些数据点违反间隔规则，即它们可以位于间隔内部或甚至在超平面的对面。这是通过引入松弛变量$\xi_i$来实现的，它们表示第$i$个数据点违反间隔的程度。

    软间隔SVM的优化问题可以表示为：

    $$ \min_{\mathbf{w}, b, \xi} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^n \xi_i $$

    受到以下约束：

    $$ y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1 - \xi_i, \quad \forall i $$
    $$ \xi_i \geq 0, \quad \forall i $$

    其中，$C$是一个正则化参数，控制了间隔违规的惩罚程度。较大的$C$值意味着对间隔违规的惩罚更重，可能导致过拟合；较小的$C$值则允许更多的间隔违规，可能提高模型的泛化能力，但也可能导致欠拟合。

6. **正则化（Regularization）**：为了防止过拟合，SVM通过正则化项（通常是权重的L2范数）来控制模型的复杂度。

SVM的优化问题可以表示为一个凸二次规划问题，这保证了找到的解是全局最优解。SVM在许多实际应用中表现出色，包括图像识别、生物信息学、文本分类等领域。

### 学习表征
认为特征本身应该被学习。
### AlexNet
模型设计：通道数多（比LeNet多10倍）；双数据流设计，每个 GPU 负责一般的存储和计算模型的工作。

激活函数：ReLU。更简单，不会出现梯度消失。

容量控制和预处理：暂退法。训练时增加了大量的图像增强数据（反转、平移等），更健壮，减少过拟合。

```py
import torch
from torch import nn
from d2l import torch as d2l

net = nn.Sequencial(nn.Conv2d(1, 96, kernel_size=5, stride=4, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=5, padding=2), nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2), nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(), nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(), nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(), nn.Linear(6400, 4096), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(4096, 10))

X = torch.randn(1, 1, 224, 224)
for layer in net:
    X=layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)
```

### 读取数据集
由于原始分辨率 28\*28 小于 AlexNet 所需要的，因此把它增加到 224\*224（尽管不合法）。
```py
batch_size=128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
```

### 训练
```py
lr, nuim_epochs = 0.01, 10
d2l.train_ch6(net, train_iter, test_iter, num_epochs, l2, d2l.try_gpu())
```

## 使用块的网络（VGG）
### VGG 块
经典卷积神经网络的基本组成部分是下面序列：
1. 带填充以保持分辨率的卷积层
2. 非线性激活函数，如 ReLU
3. 汇聚层，池化
和一个VGG块类似
```py
# import stuff
def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)       # trick

conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(*conv_blks, nn.Flatten(), nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5), nn.Linear(4096, 10))

net = vgg(conv_arch)

X = torch.rand(size=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__, 'output shape:\t', X.shape)
```

### 训练模型
构建一个通道数较少的网络，足够用于训练 Fashion-MNIST
```py
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)

lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

## 网络中的网络 NiN
其中加入两个 kernel_conv = 1*1 的层，充当 ReLU的逐像素全连接层。
```py
# import stuff
def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())
```
NiN 和 AlexNet 之间的一个显著区别是 NiN 完全取消了全连接层。

用一个NiN块，输出通道数=类别数量，最后放全局平均汇聚层，生成对数几率。减少参数数量。
```py
net = nn.Sequential(nin_block(1, 96, kernel_size=11, strides=4, padding=0),
        nn.MaxPool2d(3, stride=2),
        nin_block(96, 256, kernel_size=5, strides=1, padding=2),
        nn.MaxPool2d(3, stride=2),
        nin_block(256, 384, kernel_szie=3, strides=1, padding=1),
        nn.MaxPool2d(3, stride=2),
        nn.Dropout(0.5),
        nin_block(384, 10, kernel_size=3, strides=1, padding=1),  # label_num = 10
        nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())

X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)
```
#### AdaptiveAvgPool2d
自适应平均池化（Adaptive Average Pooling）是一种深度学习中的操作，它可以将任意大小的输入特征图转换为固定大小的输出特征图。其工作原理是通过计算输入特征图的尺寸和输出尺寸，动态地确定池化核的大小和步长，而不是使用固定的池化窗口。这种池化层对于处理不同尺寸的输入非常有用，特别是在需要将特征图调整到特定尺寸以匹配全连接层或其他层的大小时。

```py
# training
lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

Notes：全连接层就是最后的线性层，1024\*1 -> 128\*1 -> 16\*1 -> ... 这些

• NiN使用由一个卷积层和多个1 × 1卷积层组成的块。该块可以在卷积神经网络中使用，以允许更多的每像素非线性。

• NiN去除了容易造成过拟合的全连接层，将它们替换为全局平均汇聚层（即在所有位置上进行求和）。该汇聚层通道数量为所需的输出数量（例如， Fashion‐MNIST的输出为10）。

• 移除全连接层可减少过拟合，同时显著减少NiN的参数。

• NiN的设计影响了许多后续卷积神经网络的设计。

## 含并行连接的网络 GoogLeNet

在GoogLeNet中，基本的卷积块被称为Inception块（Inception block）。

Inception块由四条并行路径组成。前三条路径使用窗口大小为1 × 1、 3 × 3和5 × 5的卷积层，
从不同空间大小中提取信息。中间的两条路径在输入上执行1 × 1卷积，以减少通道数，从而降低模型的复杂性。第四条路径使用3 × 3最大汇聚层，然后使用1 × 1卷积层来改变通道数。这四条路径都使用合适的填充来使输入与输出的高和宽一致，最后我们将每条线路的输出在通道维度上连结，并构成Inception块的输出。在Inception块中，通常调整的超参数是每层输出通道数。

```py
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

class Inception(nn.Module):
    # c1~c4 是每条路径输出的通道数
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3)
        self.p3_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1, padding=1)
        self.p3_2 = nn.Conv2d(c2[0], c2[1], kernel_size=5, padding=2)
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)
    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # 在通道维度上连结输出
        return torch.cat((p1, p2, p3, p4), dim=1)

# the net
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
    nn.ReLU(),
    nn.Conv2d(64, 192, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

# 第一个Inception块的输出通道数为64 + 128 + 32 + 32 = 256，四个路径之间的输出通道数量比为64 : 128 : 32 : 32 = 2 : 4 : 1 : 1。以下的inception都类似计算。
b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
    Inception(256, 128, (128, 192), (32, 96), 64),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
    Inception(512, 160, (112, 224), (24, 64), 64),
    Inception(512, 128, (128, 256), (24, 64), 64),
    Inception(512, 112, (144, 288), (32, 64), 64),
    Inception(528, 256, (160, 320), (32, 128), 128),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
    Inception(832, 384, (192, 384), (48, 128), 128),
    nn.AdaptiveAvgPool2d((1,1)),
    nn.Flatten())

net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))

# RUN

X = torch.rand(size=(1, 1, 96, 96))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)

# Train
lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```
• Inception块相当于一个有4条路径的子网络。它通过不同窗口形状的卷积层和最大汇聚层来并行抽取
信息，并使用1×1卷积层减少每像素级别上的通道维数从而降低模型复杂度。

• GoogLeNet将多个设计精细的Inception块与其他层（卷积层、全连接层）串联起来。其中Inception块的通道数分配之比是在ImageNet数据集上通过大量的实验得来的。

• GoogLeNet和它的后继者们一度是ImageNet上最有效的模型之一：它以较低的计算复杂度提供了类似
的测试精度。

## Batch normalization
批量规范化应用于单个可选层（也可以应用到所有层），其原理如下：在每次训练迭代中，我们首先规范化输入，即通过减去其均值并除以其标准差，其中两者均基于当前小批量处理。接下来，我们应用比例系数和比例偏移。正是由于这个基于批量统计的标准化，才有了批量规范化的名称。

如果我们尝试使用大小为1的小批量应用批量规范化，我们将无法学到任何东西。这是因为在减去均
值之后，每个隐藏单元将为0。所以，只有使用足够大的小批量，批量规范化这种方法才是有效且稳定的。

从形式上来说,用 $x\in\mathcal{B}$ 表示一个来自小批量 $\mathcal{B}$ 的输入,批量规范化 BN根据以下表达式转换 x:

$$BN(x)=\gamma\odot\frac{x-\hat{\mu}_{\mathcal{B}}}{\hat{\sigma}_{\mathcal{B}}}+\beta.\qquad(7.5.1)$$

在(7.5.1)中, $\hat{\mu}_{\mathcal{B}}$ 是小批量 $\mathcal{B}$ 的样本均值, $\hat{\sigma}_{\mathcal{B}}$ 是小批量 $\mathcal{B}$ 的样本标准差。应用标准化后,生成的小批量的平均值为 0和单位方差为 1。由于单位方差(与其他一些魔法数)是一个主观的选择,因此我们通常包含拉伸参数(scale) $\gamma$ 和偏移参数(shift) $\beta$,它们的形状与 x相同。请注意, $\gamma$ 和 $\beta$ 是需要与其他模型参数一起学习的参数。由于在训练过程中,中间层的变化幅度不能过于剧烈,而批量规范化将每一层主动居中,并将它们重新调整为给定的平均值和大小 $\left(\right.$ 通过 $\left.\hat{\mu}_{\mathcal{B}}\right.$ 和 $\left.\hat{\sigma}_{\mathcal{B}}\right)$。

从形式上来看,我们计算出(7.5.1)中的 $\hat{\mu}_{\mathcal{B}}$ 和 $\hat{\sigma}_{\mathcal{B}}$,如下所示:

$$
\begin{align*}
&\hat{\mu}_{\mathcal B}=\frac{1}{|\mathcal B|}\sum_{x\in\mathcal B} x,\\
&\hat{\sigma}_{\mathcal B}^2=\frac{1}{|\mathcal B|}\sum_{x\in\mathcal B}\left(x-\hat{\mu}_{\mathcal B}\right)^2+\epsilon.
\end{align*}
\qquad(7.5.2)
$$

请注意,我们在方差估计值中添加一个小的常量 $\epsilon>0$, 以确保我们永远不会尝试除以零

卷积层：对每个通道都进行规范化。全连接层：先规范化后激活函数。

### Starting from scratch
```py
# import

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # 通过is_grad_enabled来判断当前模式是训练模式还是预测模式
    if not is_grad_enabled():
        # 如果是在预测模式下，直接使用传入的移动平均所得的均值和方差
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 使用全连接层的情况，计算特征维上的均值和方差
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # 使用二维卷积层的情况，计算通道维上（axis=1）的均值和方差。
            # 这里我们需要保持X的形状以便后面可以做广播运算
            mean = X.mean(dim=(0, 2, 3), keepdim=True)          
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # 训练模式下，用当前的均值和方差做标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 更新移动平均的均值和方差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta # 缩放和移位
    return Y, moving_mean.data, moving_var.data
```
#### 有关 moving_mean, moving_var:
在每一批次（batch）的训练过程中，Batch Normalization 会计算均值 (mean) 和 标准差 (variance)，用于对输入数据进行归一化。

在推理（inference）阶段，模型不再接受小批量数据作为输入，而是一个单一的数据点。因此，Batch Normalization 需要依赖训练过程中累积的统计信息，而不是当前小批量的均值和方差。

注意！batch normalization 是在同一个 batch 内所有输入矩阵 X 对应位置 $a_{ij}^{(t)}, t=1, 2, ...$ 求 mean, variance，而不是在单个数据矩阵 X 的内部求。

**具体来说**，Batch Normalization 对于每个通道上的特定位置（即每个空间位置）的像素值进行归一化处理。具体来说，Batch Normalization 在每个通道上独立计算整个批量中所有图像的均值和方差，然后使用这些统计量来规范化该通道的每个像素值。

这里的“特定位置”指的是特征图（feature map）中的空间位置，即高度（H）和宽度（W）的每个点。对于每个通道，Batch Normalization 会计算所有图像在该空间位置上的像素值的均值和方差，然后使用这些值来调整和缩放该位置的像素值，使得调整后的像素值具有零均值和单位方差。

**再进一步说** 在卷积神经网络（CNN）中，一个图像在某个卷积层之后确实会生成多个通道（channels）。这些通道是特征图（feature maps）的集合，每个特征图代表不同的特征或激活。

当应用Batch Normalization时，不是将所有图像的所有通道加起来，而是对每个通道独立地进行归一化处理。具体来说：

1. **批量维度（N）**：Batch Normalization会考虑批量中的所有图像。这意味着对于每个通道，它会计算批量中所有图像在该通道上所有空间位置的均值和方差。

2. **通道维度（C）**：对于每个通道，Batch Normalization会独立地计算均值和方差。这意味着每个通道的归一化参数（均值和方差）是独立的。

3. **空间维度（H和W）**：对于每个通道中的每个空间位置（即每个像素位置），Batch Normalization会使用整个批量中该位置的像素值来计算均值和方差。

因此，Batch Normalization是在批量中的每个图像的每个通道上进行的，它考虑了批量大小、通道数以及空间维度（高度和宽度），但归一化过程是在通道维度上独立进行的。这样做的目的是为了保持每个通道特征的分布一致性，同时允许不同通道之间有不同分布，因为它们可能代表不同的特征。


```py
class BatchNorm(nn.Module):
    # num_features：完全连接层的输出数量或卷积层的输出通道数。
    # num_dims： 2表示完全连接层， 4表示卷积层
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成1和0
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 非模型参数的变量初始化为0和1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)
    def forward(self, X):
        # 如果X不在内存上，将moving_mean和moving_var
        # 复制到X所在显存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 保存更新过的moving_mean和moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        return Y


# put into use: a simpler perspective
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(16*4*4, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
    nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(),
    nn.Linear(84, 10))

lr, num_epochs, batch_size = 1.0, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```
#### 为什么对于 γ, β 要进行反向传播？
Batch Normalization 的核心思想是对每一层的输入进行标准化，使得该层的输出均值接近0，方差接近1。这样做的目的是减少内部协变量偏移（internal covariate shift），从而加速训练并提高网络的稳定性。

但是，标准化后的数据可能会丧失一些原本有意义的特征。例如，假设输入数据原本的分布对模型有帮助，直接归一化可能会让这些特征变得不那么显著。因此，Batch Normalization 会引入 缩放（scale, γ） 和 平移（shift, β） 参数，来恢复这种灵活性。

- γ：用于缩放归一化后的输出。通过调整 γ，网络可以控制输出的方差。
- β：用于平移归一化后的输出。通过调整 β，网络可以控制输出的均值。

反向传播的目的是通过梯度下降（或其他优化方法）来最小化损失函数，从而优化网络的参数。Batch Normalization 中，**γ** 和 **β** 是训练过程中需要优化的参数，因为它们直接影响网络的输出分布。

1. **对 γ 的梯度**：
   γ 是乘以归一化结果 $\hat{x}$ 的系数。在反向传播时，网络计算损失函数相对于网络输出的梯度，这个梯度会被传递到 $\hat{x}$。然后，通过链式法则，$\gamma$ 会对最终的梯度产生影响。通过更新 $\gamma$，网络能够调整归一化后输出的尺度，从而改善训练过程。

   $$
   \frac{\partial L}{\partial \gamma} = \frac{\partial L}{\partial y} \cdot \hat{x}
   $$

2. **对 β 的梯度**：
   β 是加到归一化结果 $\hat{x}$ 上的常数项，它会影响网络输出的偏移量。在反向传播中，β 会影响最终的输出，因此需要根据损失函数的梯度对其进行更新。

   $$
   \frac{\partial L}{\partial \beta} = \frac{\partial L}{\partial y}
   $$

   其中，$\frac{\partial L}{\partial y}$ 是损失函数相对于归一化输出 $y$ 的梯度。

##### 总结
- **缩放参数（γ）** 和 **平移参数（β）** 是在 Batch Normalization 中引入的额外参数，目的是恢复标准化过程中丧失的特征，使得网络能够自适应地调整输出的分布。
- 在 **反向传播** 时，这两个参数会参与梯度计算和更新，因为它们直接影响神经网络的输出。通过优化这些参数，网络能够更好地适应数据分布，提升训练效果和稳定性。

### 小结
- 在模型训练过程中，批量规范化利用小批量的均值和标准差，不断调整神经网络的中间输出，使整个神经网络各层的中间输出值更加稳定。
- 批量规范化在全连接层和卷积层的使用略有不同。
- 批量规范化层和暂退层一样，在训练模式和预测模式下计算不同。
- 批量规范化有许多有益的副作用，主要是正则化。另一方面，”减少内部协变量偏移“的原始动机似乎不是一个有效的解释。

## 残差网络 ResNet
#### 寻找函数：
现在假设$f^*$是我们真正想要找到的函数，如果是 $f^* \in \mathcal{F}$，那我们可以轻而易举的训练得到它，但通常我们不会那么幸运。相反，我们将尝试找到一个函数$f^*_\mathcal{F}$ ，这是我们在F中的最佳选择。
$$
f_{\mathcal{F}}^* := \arg\min_{f} L(\mathbf{X}, \mathbf{y}, f) \text{ subject to } f \in \mathcal{F}.
$$
不断找更复杂的函数，同时只有当较复杂的函数类包含较小的函数类时，我们才能确保提高它们的性能。

对于深度神经网络，如果我们能将新添加的层训练成恒等映射（identity function） f(x) = x，新模型和原模型将同样有效。这就是我们的目标。

#### 残差网络的核心思想：
每个附加层都应该更容易地包含原始函数作为其元素之一。

假设我们的原始输入为x，而希望学出的理想映射为f(x)。一个正常块需要直接拟合出该映射f(x)，而一个残差块需要拟合出残差映射 h(x) = f(x) − x （学这个更简单）， 然后再从输入的地方获得 x ，将 h 和 x 相加从而得到 f(x)（就是shortcut）。这样即使从这个块中没学到东西（f(x) - x 不好），也不会偏差过大，还能从shortcut直接把上一层学到的函数（也就是现在的输入x）传递过去，使得新的复杂模型还是能包含先前的简单模型的。最后作为激活函数的输入。

```py
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:  # 残差块的function，调整通道和分辨率
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

# 输入输出形状一致
blk = Residual(3,3)
X = torch.rand(4, 3, 6, 6)
Y = blk(X)
Y.shape

# 我们也可以在增加输出通道数的同时，减半输出的高和宽。
blk = Residual(3,6, use_1x1conv=True, strides=2)
blk(X).shape
```

### ResNet model
```py
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

# 4个由残差块组成的模块，每个模块使用若干个同样输出通道数的残差块。
# 第一个模块的通道数同输入通道数一致。由于之前已经使用了步幅为2的最大汇聚层，所以无须减小高和宽。
# 之后的每个模块在第一个残差块里将上一个模块的通道数翻倍，并将高和宽减半。

def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

net = nn.Sequential(b1, b2, b3, b4, b5,
nn.AdaptiveAvgPool2d((1,1)),
nn.Flatten(), nn.Linear(512, 10))

lr, num_epochs, batch_size = 0.05, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

## 稠密连接网络 DenseNet
借用了泰勒展开的定义。$f(x)=x+g(x)$，分为一个简单的线性项和一个复杂的非线性项。

DenseNet 不是使用 ResNet 的相加方式，而是用连接（[,]）方式。在应用越来越复杂的函数序列之后，我们执行从x到其展开式的映射：
$$
\textbf{x} \rightarrow \left[ \textbf{x}, f_1(\textbf{x}), f_2(\left[ \textbf{x}, f_1(\textbf{x}) \right]), f_3(\left[ \textbf{x}, f_1(\textbf{x}), f_2(\left[ \textbf{x}, f_1(\textbf{x}) \right]) \right]), \ldots \right].
$$
最后连接进去。

```py
import torch
from torch import nn
from d2l import torch as d2l

def conv_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))

class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(
                num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)
    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # 连接通道维度上每个块的输入和输出
            X = torch.cat((X, Y), dim=1)
        return X

blk = DenseBlock(2, 3, 10)
X = torch.randn(4, 3, 8, 8)
Y = blk(X)
print(Y.shape)
```
### 过渡层
7.7.3 过渡层
由于每个稠密块都会带来通道数的增加，使用过多则会过于复杂化模型。而过渡层可以用来控制模型复杂度。
它通过1 × 1卷积层来减小通道数，并使用步幅为2的平均汇聚层减半高和宽，从而进一步降低模型复杂度。
```py
def transition_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2))
```

### DenseNet
```py
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

# num_channels为当前的通道数
num_channels, growth_rate = 64, 32
num_convs_in_dense_blocks = [4, 4, 4, 4]
blks = []
for i, num_convs in enumerate(num_convs_in_dense_blocks):
    blks.append(DenseBlock(num_convs, num_channels, growth_rate))
    # 上一个稠密块的输出通道数
    num_channels += num_convs * growth_rate
    # 在稠密块之间添加一个转换层，使通道数量减半
    if i != len(num_convs_in_dense_blocks) - 1:
        blks.append(transition_block(num_channels, num_channels // 2))
        num_channels = num_channels // 2

net = nn.Sequential(
    b1, *blks,
    nn.BatchNorm2d(num_channels), nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(num_channels, 10))

lr, num_epochs, batch_size = 0.1, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

```