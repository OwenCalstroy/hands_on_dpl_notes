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