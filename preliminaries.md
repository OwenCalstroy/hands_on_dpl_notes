# PRELIMINARIES

## np ###
```py
import torch 
x = torch.arange(12)
print(x)
print(x.shape)

x.numel()

x = x.reshape(3, 4)

zeros = torch.zeros((2, 3, 4))
ones = torch.ones((2, 3, 4))

torch.randn(3, 4)

torch.tensor([1, 2, 3])

x = torch.tensor([1, 2, 3])
y = torch.tensor([0.1, 0.2, 0.3])
print(x+y, x-y, x*y, x/y, y ** x)

torch.exp(x)
```
concatenate vectors(here tensors )
```py
X = torch.arange(12, dtype=torch.float32).reshape(3, 4)
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
Z = torch.cat((X, Y), dim = 0)
Z_prime = torch.cat((X, Y), dim = 1)
```
bool array
```py
X == Y 
```
sum
```py
X.sum()
```
broadcast: copy the vector
```py
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2)
print(a + b)
```
id() -> shows the position in ram
```py
before = id(Y)
Y = Y + X 
id(Y) == before #False
```

Y[:] = ... function locally/ += function also
```py
Y[:] = Y + X 
```

numpy <=> torch
```py
A = X.numpy()
B = torch.tensor(A)
```

## pandas ###
```py
import os 
#write in doc
os.makedirs(os.path.join('C:/Users/weiyi/Desktop/hands_on_dp_code', 'data'), exist_ok=True) #注意更改位置
data_file = os.path.join('C:/Users/weiyi/Desktop/hands_on_dp_code', 'data', 'house_tiny.csv') #comma separate value
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n') #colomn name
    f.write('NA, Pave, 127500\n')
    f.write('2, NA, 1060000\n')
    f.write('4, NA, 178100\n')
    f.write('NA, NA, 140000\n')

import pandas as pd 
data = pd.read_csv(data_file)
print(data)
```

insertion or deletion to deal with missing data
using .iloc to separate the csv
```py
inputs, outputs = data.iloc[:,0:2], data.iloc[:, 2:]
inputs = inputs.fillna(inputs.mean())                                       #something wrong
print(inputs)
```

get_dummies -> get categories: in a certain one or not (0/1 or T/F) + the 'NaN' category
```py
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
```

turn to vector: .to_numpy
'import torch' first
```py
x = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(outputs.to_numpy(dtype=float))
print(x, y)
```

## pytorch differentiation
```py
import torch
x = torch.arange(4.0)
x.requires_grad_(True)
```
最后一行 <=> torch.arange(4.0, requires_grad=True)
requires_grad 让反向传播时计算并存储这个张量的梯度，False 时不存储节约内存

    默认为 False，注意要改成 True 否则会报错

grad_fn 是张量的属性，指向创建该张量的函数。用在反向传播时追踪梯度的计算图
```py
y = x * x
print(y)
y = 3 * x.dot(x.T) + 2
print(y)
```

### backward()
自动求导：当你在PyTorch中创建一个张量并设置requires_grad=True时，这个张量会跟踪在其上执行的所有操作，形成一个计算图。调用.backward()方法时，系统会沿着这个计算图反向传播，根据链式法则计算每个叶子节点（即最初具有requires_grad=True属性的输入张量）对当前目标变量（通常是损失函数值）的梯度

标量与非标量输出：如果损失值是一个标量（单个数值），则直接调用loss.backward()即可。如果损失值是一个向量或矩阵，则需要指定一个gradient参数来指明对每个元素求导的权重，这是因为backward()函数默认只能对标量进行操作

梯度累积：每次调用backward后，梯度会被累积到.grad属性中，而不是被覆盖。因此，如果你不想累积梯度，需要在每次调用backward前调用.zero_()方法清除梯度

    累积许多小批量的梯度，然后最后一起更新。所以对于一个大 batch 里面的小 batch 不需要清除梯度。

    如果不清空，那么多次得到的结果会进行对应位置的相加：
    [1, 2, 3, 4] & [5, 6, 7, 8] -> [6, 8, 10, 12]

#### 参数详解：

1. gradient（可选）：如果tensor不是标量，则需要传递一个与这个张量形状相同的gradient参数，指明对每个元素的梯度贡献。
2. retain_graph（布尔值，默认为False）：如果设置为True，则保留计算图，用于多次调用backward。这在某些情况下很有用，比如当你需要在一个模型的不同部分多次计算梯度时。
3. create_graph（布尔值，默认为False）：如果设置为True，则对梯度计算图进行记录，以便计算更高阶的梯度。

这里的梯度贡献/权重也就是：你得到的偏导数*对应位置的数
```py
x = torch.arange(4.0)
x.requires_grad_(True)
y = x * x
y.backward(torch.tensor([1, 1, 1, 1])) 
print(x.grad) # -> [2*x1, 2*x2, 2*x3, 2*x4]
y.backward(torch.tensor([1, 1, 1, 2]))
print(x.grad) # -> [2*x1, 2*x2, 2*x3, 2*x4 * 2]
```

在默认情况下，Pytorch 会累计梯度 -> 清楚之前的值
```py
x.grad.zero_()
y = x.sum()
y.backward()
x.grad
```

    矩阵乘法在 Pytorch 里用 torch.matmul() / A @ B 实现

对于 y = x * z, 在y反向传播之后可以分别得到 x、z 的导数
```py
import torch
x = torch.arange(1.0, 5.0)
x.requires_grad_(True)
z = torch.arange(5.0, 9.0)
z.requires_grad_(True)
y = z * x
y.backward(torch.tensor([1, 1, 1, 1]))
print(x.grad, z.grad)
```

### 分离计算
y(x), z(y, x), 但不希望 dz/dx 的时候把 dy/dx 算进去，要把y当常数：detach()
```py
x.grad.zero_() # reset the code before
y = x * x
u = y.detach() # 此时 u和 y有相同的值，但丢弃计算途中如何计算 y的任何信息。梯度不会向后流经 u到 x
z = u * x
z.sum().backward()
x.grad == u # True

x.grad.zero_()
y.sum().backward() # 超好用
x.grad == 2 * x # True
```

在运行反向传播函数后，立即再次运行，会报错：Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward.

## 概率
#### set up
```py
import matplotlib.pyplot as plt
import torch
from torch.distributions import multinomial
from d2l import torch as d2l
```

传入概率向量，输出另一个长度相同的向量：索引i处的值是采样结果中i出现的次数
```py
fair_probs = torch.ones([6]) / 6
print(multinomial.Multinomial(1, fair_probs).sample()) # 1是次数
# >>> something like 'tensor([0, 0, 1, 0, 0, 0])

# true possibilities simulation:
print(multinomial.Multinomial(10000, fair_probs).sample() / 10000)

# show plotly
counts = multinomial.Multinomial(10, fair_probs).sample((500,)) # 500 groups of experiment
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)

d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].numpy(), label=("P(die=" + str(i
    +1) + ")"))
    d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
    d2l.plt.gca().set_xlabel('Groups of experiments')
    d2l.plt.gca().set_ylabel('Estimated probability')
    d2l.plt.legend()
plt.show()
```



