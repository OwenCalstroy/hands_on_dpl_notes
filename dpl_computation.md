# Deep Learning Computation
## 层和块
神经网络块。块 block 可以描述单个层、由多个层组成的组件或者整个模型本身。可以将块组合成更大的组件（递归）。

块是类 class。需要 forward 函数，存储必须的参数。

net(X) is actually the abbreviation of net.__call__(X)
### 自定义块
块需要配提供的基本功能：
1. 将输入数据作为其前向传播函数的参数。
2. 通过前向传播函数来生成输出。请注意，输出的形状可能与输入的形状不同。例如，我们上面模型中的 第一个全连接的层接收一个20维的输入，但是返回一个维度为256的输出。
3. 计算其输出关于输入的梯度，可通过其反向传播函数进行访问。通常这是自动发生的。
4. 存储和访问前向传播计算所需的参数。
5. 根据需要初始化模型参数。
```python
class MLP(nn.Module):
    # 用模型参数声明层。这里声明两个全连接的层
    def __init__(self):
        # 调用MLP的父类Module的构造函数来执行必要的初始化。
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数params
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    # 定义模型的向前传播，即如何根据输入X返回所需的模型输出
    def forward(self, X):
        # 这里使用ReLU的函数版本，在nn.functional模块里定义
        return self.out(F.relu(self.hidden(X)))
```

### 顺序块
Sequential 的设计是为了把其他模块串起来。为了构建我们自己的简化的 MySequential，我们只需要定义两个关键函数:
1. 一种将块逐个追加到列表中的函数;
2. 一种前向传播函数，用于将输入按追加块的顺序传递给块组成的“链条”。
```py
class MySequential(nn.Module):
    def __init__(self,*args):
        super().__init__()
        for idx, module in enumerate(args):
            # module是Module子类的一个实例。保存在‘Module’类的成员变量_modules里。_module的类型是OrderedDict
            self._modules[str(idx)] = module
    
    def forward(self, X):
        # OrderedDict 保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X = block(X)
        return X        # 牛逼啊这样简单
```
### 在向前传播函数中执行代码
有时希望合并既不是上一层的结果也不是可更新参数的项。
```py
class FixedHiddenMLP(nn.Moduel):
    def __init__(self):
        super().__init__()
        # 不计算梯度的随即权重参数。因为其在训练期间保持不变
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)
    def forward(self, X):
        X = self.linear(X)
        # 使用创建的常量参数以及relu和mm函数
        X = F.relu(torch.mm(X, self.rand_weight) + 1) # matmul
        # 复用全连接层。这相当于两个全连接层共享参数
        X = self.linear(X)
        # control flow
        hwile X.abs().sum() > 1:
        X /= 2
        reutnr X.sum()
```
这个 X 权重不是一个模型参数，永远不会被反向传播更新。

## 参数管理
```py
import torch 
from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
net(X)
```
### 参数访问
```py
print(net[2].state_dict()) # >>> 'weight' & 'bias'

print(type(net[2].bias))
print(net[2].bias)  # 还有 requires_grad 项
print(net[2].bias.data)

net[2].weight.grad==None #>>> True

# visit the params
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])

net.state_dict()['2.bias'].data
```
'*' 操作符用于解包（unpacking）列表或元组。具体来说，它将括号内的对象（在这个例子中是一个列表）解包成独立的参数，然后传递给 print 函数。

这行代码的工作原理如下：

1. [(name, param.shape) for name, param in net[0].named_parameters()] 是一个列表推导式（list comprehension），它遍历 net[0].named_parameters() 返回的迭代器。named_parameters() 方法返回模型参数的名称和值的元组迭代器。

2. 对于每一对（名称，参数），列表推导式创建一个新的元组，其中包含参数的名称和形状。

3. '*' 操作符将列表中的每个元组解包，并将元组内的元素作为独立的参数传递给 print 函数。

4. print 函数随后打印这些参数，每个参数在新的一行上。
#### 从嵌套块收集参数 —— 精华之处
将多个块相互嵌套，参数命名约定是如何工作？
```py
def block1():
    return nn.Sequential(nn.linear(4, 8), nn.ReLU(), nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequencial()
    for i in range(4):
        # 在这里嵌套
        net.add_module(f'block {i}', block1())          # 修改名称中的数字用这个方法！方便快捷
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
rgnet(X)

rgnet[0][1][0].bias.data
```

### 参数初始化
内置初始化
```py
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)
net.apply(init_normal) #apply 是全局的
net[0].weight.data[0], net[0].bias.data[0]

# 还可用 nn.init.constant_(target)

# 还可以对某些块应用不同的初始化方法
def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)
net[0].apply(init_xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)
```
自定义初始化

使用以下方法：
$$ w = \begin{cases} 
U(5, 10) & \text{possibility } 0.25 \\
0 & \text{possibility } 0.5 \\
U(-10, -5) & \text{possibility } 0.25 \\
\end{cases} $$
```py
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape) for name, param in m.named_parameters()][0]) # only weight
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5

net.apply(my_init)
net[0].weight[:2]
```
面对可能性问题，利用  随机数+划定区间范围 来达到效果。

可以直接设置参数：
```py
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
net[0].weight.data[0]
```







