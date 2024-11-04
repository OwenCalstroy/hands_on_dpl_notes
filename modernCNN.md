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














