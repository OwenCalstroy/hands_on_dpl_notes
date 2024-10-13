# 线性网络
## 线性回归
### 矢量化加速
settings below
```py
import matplotlib.pyplot as plt
import math
import time
import numpy as np
import torch
for d2l import torch as d2l
```
测试工作负载
```py
class Timer:    #@save
    """记录多次运行时间"""
    def __init__(self):
        self.times = []
        self.start()         # 调用的时候就开始计时了！

    def start(self):
        """start timer"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]
    
    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()
    
c = torch.zeros(n)
timer = Timer()
for i in range(n):
    c[i] = a[i] + b[i]
print(f'{timer.stop():.5f} sec')
```

    Tips:

    python 里的初始化在调用的时候就启动，所以可以把一开始就要启动的function放进去

而矢量化代码能巨大加速！

### 正态分布与平方损失
可视化正态分布
```py
def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(-0.5 / sigma**2 * (x - mu)**2)

#可视化
x = np.arange(-7, 7, 0.01)

#均值与标准差对
params = [(0, 1), (0, 2), (3, 1)]
d2l.plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x', ylabel='p(x)', figsize=(4.5, 2.5), legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])  # 牛逼啊用数组作为整体赋值
plt.show()

```

## 线性回归的实现
y = wx + b + e, b 为常数，e 为噪音，呈正态分布

#### 语法：

torch.normal(mean, std, *size) std 标准差
torch.matmul(A, B) 矩阵乘法
.reshape((-1, 1)) 里-1表示该维度的大小会自动计算，从而保证张量的总元素保持不变

```py
def synthetic_data(w, b, num_examples):
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape) # noise
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, ture_b, 1000)

print('features:', features[0],'\nlabel:', labels[0])

d2l.set_figsize()
d2l.plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1)
plt.show()
```

这里有几个原因为什么需要使用 .detach()：

    避免梯度计算：如果你不使用 .detach()，当你尝试将张量转换为NumPy数组时，PyTorch会报错，因为它默认情况下会保留梯度信息。这是因为PyTorch的张量默认是计算图的一部分，而计算图是用于自动梯度计算的。

    内存消耗：保留梯度信息会占用额外的内存。如果你不需要进行梯度计算，那么保留这些信息就没有意义，使用 .detach() 可以减少内存消耗。

    兼容性：某些操作和库（比如matplotlib）可能不直接支持PyTorch张量，而是需要NumPy数组。使用 .detach() 可以将张量转换为NumPy数组，从而兼容这些操作和库。

### 读取数据集 转换小批量
```py
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indicies = list(range(num_examples))
    # randomly read the samples
    random.shuffle(indicies)
    for i in range(0, num_examples, batch_size):
        batch_indicies = torch.tensor(indicies[i:min(i + batch_size, num_examples)])
        yield features[batch_indicies], labels[batch_indicies]

batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break
```
学习这样的切割方法：(同时注意最后min函数的选择收尾)
```
for i in range(0, num_examples, batch_size):
    batch_indicies = torch.tensor(indicies[i:min(i + batch_size, num_examples)])
```
random.shuffle(): 在Python中，random.shuffle 是一个用于随机打乱序列的函数。它属于 random 模块，可以对列表（list）进行原地（in-place）的随机排序。这意味着它直接修改传入的列表，而不是返回一个新的列表。

注意这里使用了 yield 函数，在每次调用的时候都回到循环，进行下一次循环然后 return。

### 初始化模型参数
```py
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_gread=True)
```

### 定义模型、损失函数、梯度下降
```py
def linreg(X, w, b):
    """线性回归模型"""
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
```
with torch.no_grad() 的使用：

    在PyTorch中，torch.no_grad() 是一个上下文管理器，用于暂时禁用梯度计算。这在处理推理（inference）或评估（evaluation）阶段非常有用，因为在这些阶段不需要进行反向传播，因此可以节省内存和计算资源。

    使用 torch.no_grad() 可以减少模型在前向传播时的内存占用，因为梯度信息不会被存储。此外，它还可以减少计算量，因为不需要计算梯度。

注意要归零param.grad

lr -> learning rate

### 训练
```py
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y) # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。l中所有元素加到一起计算梯度（trick）
        l.sum().backward()
        sgd([w, b], lr, batch_size) # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
```
在机器学习和深度学习中，一个“epoch”指的是对整个训练数据集进行一次完整的遍历。换句话说，就是将训练集中的所有数据都用于模型训练一次。在训练过程中，模型会学习到数据中的模式和关系，以期望能够对新的、未见过的数据做出准确的预测。

以下是一些与“epoch”相关的要点：

1. **训练过程**：在训练一个模型时，通常会将数据集分成多个批次（batches），每个批次包含一定数量的样本。模型会对每个批次的数据进行学习，更新其参数，然后移动到下一个批次。当模型遍历完所有批次后，就完成了一个epoch。

2. **迭代次数**：一个epoch通常包含多个迭代（iterations），每个迭代对应于一个批次的数据。因此，迭代次数等于训练数据集中的样本总数除以每个批次的大小。

3. **模型性能**：在每个epoch结束后，通常会在验证集上评估模型的性能，以监控模型是否过拟合（overfitting）或欠拟合（underfitting）。如果模型在训练集上的性能很好，但在验证集上的性能不佳，这可能意味着模型过拟合了。

4. **早停法（Early Stopping）**：为了防止过拟合，可以使用早停法。在训练过程中，如果在连续多个epoch中验证集上的性能没有显著提升，可以提前停止训练，以避免进一步的过拟合。

5. **学习率调整**：在训练过程中，可以调整学习率（learning rate），以帮助模型更好地收敛。例如，可以在训练的后期降低学习率，以进行更精细的参数调整。

6. **训练时长**：epoch的数量是训练时长的一个决定因素。训练更多的epoch可能会提高模型的性能，但同时也会增加训练时间和计算成本。因此，需要在模型性能和训练资源之间找到一个平衡点。

在实际应用中，epoch的数量和每个epoch的迭代次数是超参数，需要根据具体问题和数据集来调整。通常，这些超参数是通过交叉验证（cross-validation）和网格搜索（grid search）等方法来选择的。

## 线性回归的简单实现
### generate dataset
```py
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
```

### read dataset
```py
def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)
print(next(iter(data_iter)))
```

### define model
```py
from torch import nn
net = nn.Sequential(nn.Linear(2, 1))
```
nn.Linear(输入的特征形状，输出的特征形状)

Sequential类将多层串联在一起，就是一个神经网络

nn -> neural network

### initiate model parameters
```py
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
```
net[0] -> visit first layer

weight.data & bias.data -> visit the parameters

### loss function
```py
loss = nn.MSELoss
```
平方L2范数，返回所有样本损失的平均值

### define Optimized algorithm
```py
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
```
SGD是随机梯度下降（Stochastic Gradient Descent）的缩写

### training
```py
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
```

    针对导入数据的part还是不够清楚，记录一下，可能需要阅读源码解决


## Softmax 回归
是针对与输出 Y 是一个向量，代表着不同类型的概率/可能性

## 图像分类数据集 MNIST
setting up
We are using the similar but more complicated Fashion-MNISH dataset.
```py
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
d2l.use_svg_display()
```
### reading dataset
通过ToTensor 实例将图像数据从PIL类型变换为float32，并除以255使得所有像素的数值在0-1间。
下载FashionMNIST内置的数据集。
```py
trans = transforms.ToTensor()
minst_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True)
minst_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True)

def get_fashion_mnist_labels(labels):
    """ return Fashion-MNIST dataset's text label"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]
```
torchvision.datasets.FashionMNIST这个类的主要参数包括：

1. root：指定数据集的存储路径。如果设置 download=True，数据集将被下载到这个路径下。

2. train：一个布尔值，指定是要加载训练集（True）还是测试集（False）。

3. transform：一个可选的变换函数或变换函数的组合，用于对数据集中的图像进行预处理。例如，transforms.ToTensor() 可以将 PIL 图像或 NumPy ndarray 转换为 FloatTensor，并缩放图像像素值到 [0, 1] 范围内。

4. target_transform：（可选）一个函数或可调用对象，用于在加载数据时对目标进行变换，如对标签进行编码。

5. download：一个布尔值，指定是否需要下载数据集。如果数据集已经存在于 root 指定的路径下，可以设置为 False。

6. batch_size：（在使用 DataLoader 时指定）指定每个批次的样本数量。

FashionMNIST 里包含十个服装类别。gfml 函数用于在数字标签索引及其文本名称之间进行转换。

Visualize it!
```py
def show_images(img, num_rows, num_cols, titles=None, scale=1.5):
    """show the image list"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        # choose different ways to deal with different pic_type
        if torch.is_tensor(img):
            #picture tensor
            ax.imshow(img.numpy())
        else:
            # PIL picture
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axis.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
        return axes

```
在Python中，单个下划线 `_`（单下划线）有几种特殊的用途：

1. **临时性的忽略值**：
   单个下划线 `_` 可以作为一个临时性的占位符，用来忽略或不关注某个特定的值。例如，在迭代中，如果你只关心索引而不是元素本身，可以这样使用：

   ```python
   for _ in range(5):
       print("Hello, World!")
   ```

   在这个例子中，`_` 表示我们不关心循环的索引值。

2. **上一次操作的结果**：
   在交互式解释器（比如Python的命令行交互式环境）中，`_` 代表上一次操作的结果：

   ```python
   3 + 4
   _  # 输出: 7
   ```

   这在快速测试或计算时非常有用。

3. **国际化和本地化**：
   在涉及国际化（i18n）和本地化（l10n）的程序中，`_` 用来表示字符串应该被翻译。这是一个约定俗成的标记，一些工具（如 `gettext`）会查找以 `_` 开头和结尾的字符串，并提供翻译。

4. **私有变量的约定**：
   按照惯例，以单下划线开头的变量或函数名（如 `_internal_var`）表示它们是内部使用的，暗示着它们不应该被外部访问。但这仅仅是一种约定，Python并不会强制这种访问限制。

axis.flatten() 变为一维数组

enumerate 返回 index + value

### read small batch
```py
batch_size = 256

def get_dataloader_workers():
    """使用4个进程来读取数据"""
    return 4

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers())

# see how long should it take
timer = d2l.Timer()
for X, y in train_iter:
    continue
print(f'{timer.stop():.2f} sec')
```

### integrate all units
读取数据集， 返回训练集和验证集的数据迭代器。此外还接受 resize 调整图像大小
```py
def load_data_fashion_mnist(batch_size, resize=None):
    """下载 Fashion-MNIST 数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root='../data', train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root='../data', train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers()), data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=get_dataloader_workers()))


train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break
```
在PyTorch中，`DataLoader` 是一个迭代器，它封装了一个数据集，提供了批量加载数据的便捷方式。它通常与 `torch.utils.data.Dataset` 子类一起使用，后者负责管理数据的存储和访问。

`DataLoader` 的主要参数包括：

1. `dataset`：一个实例化的 `Dataset` 对象，包含了数据和标签。

2. `batch_size`：每个批次的样本数量。

3. `shuffle`：一个布尔值，指示是否在每个epoch开始时打乱数据。如果设置为 `True`，则数据在每个epoch迭代前会被随机打乱。

4. `num_workers`：用于指定加载数据的进程或线程数量。如果大于0，则会创建多个子进程来并行加载数据，这可以显著加速数据加载过程，特别是在处理大规模数据集时。

5. `pin_memory`：一个布尔值，如果设置为 `True`，加载的数据将被复制到CUDA固定内存中，这可以加快将数据转移到GPU的速度。

6. `drop_last`：一个布尔值，如果设置为 `True`，当数据集大小不能被 `batch_size` 整除时，最后一个不完整的批次将被丢弃。

7. `timeout`：数据加载操作的超时时间（以秒为单位）。

8. `sampler`：一个 `Sampler` 或 `SequenceSampler` 对象，用于指定如何从数据集中抽取样本。

## softmax return from ZERO
```py
import torch
from IPython import display
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

#### Initialize
```py
num_inputs = 784
num_outputs = 10

w = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = toech.zeros(num_outputs, requires_grad=True)
```

### Softmax
```py
X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(X.sum(0, keepdim=True), X.sum(1, keepdim=True))

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition

X = torch.normal(0, 1, (2, 5))
X_prob = softmax(X)
print(X_prob, X_prob.sum(1))
```

### def model
```py
def net(X):
    retrn softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)
```

### loss function
```py
y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])

def cross_entrophy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])

print(cross_entrophy(y_hat, y))
```
