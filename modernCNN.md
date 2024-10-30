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








