# HW2 Classification 

> 2025.5.12 ~ 2025.5.18

## 学习目标

1. 用 MLP 拟合一个分类模型
2. 掌握 MLP 训练基本技巧，BatchNorm, dropout
3. Using RNN/LSTM
4. Report：(1) Shallow Network vs Deep Network (2) dropout

## TODO

[x] MLP + BatchNorm + dropout
[x] LSTM
[x] 调参

### LSTM (Long Short-Term Memory) [1]

- 相比于 MLP，从“静态特征建模” → “动态时序建模”
- 相比于 RNN，从“短期递归记忆” → “长期状态控制”

LSTM 引入多个门控参数，参数多，计算开销大。

LSTM 单元在每个时间步 $t$ 维护两个状态：
- 隐藏状态 $h_t$：输出状态（传给下一层）
- 记忆状态 $c_t$：长时记忆（贯穿整个序列）

门作用：
- 遗忘门 $f_t$：决定保留多少上一个记忆 $c_{t-1}$
- 输入门 $i_t$：决定当前输入写入多少新信息
- 输出门 $o_t$：决定从当前记忆中输出多少信息

$$f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)$$

$$i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)$$
$$\tilde{c}t = \tanh(W_c x_t + U_c h{t-1} + b_c)$$

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

$$o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)$$
$$h_t = o_t \odot \tanh(c_t)$$

## Report

Deep Network 和 Shallow network 的差异：[Why Deep Learning?](https://www.bilibili.com/video/BV1Wv411h7kN/?p=33&share_source=copy_web&vd_source=e46571d631061853c8f9eead71bdb390&t=1560)

深的网络能用更少的参数产生更多的 pieces。

考虑 bias，总参量为：

$$
Totalparams = (inputdim + 1) \times hiddendim + (hiddendim + 1) \times hiddendim \times (hiddenlayers - 1) + (hiddendim + 1) \times outputdim
$$

`nn.Dropout(p=)` 在激活函数前/后增加都可以。

### tricks:

根据 hint2 以及 [phoneme](https://www.phon.ucl.ac.uk/courses/spsci/spc/lab8.html)，注意到英文中 phoneme 的持续时间都小于 11 个frames（11*25=275ms），简单粗暴设置 n_concat 为 15。

增大模型尺寸，8 layers 512 dims，kaggle public score 0.70924, private score 0.70974。

根据 hint3 设置 batchnorm 和 dropout（事实上调了调 dropout 似乎并没有什么太大用处，只会减慢拟合速度）。

对模型尺寸做精细调整，并把 n_concat 提高。

## Practice

### MLP + 拼好帧 
加入dropout，减缓了过拟合，当提高模型的深度和宽度时，模型容量增强，在拼好帧任务中很有效（concat_nframes=13，输入维度 507）。

拼接帧（MLP 传统方法）没有时间意识（无法知道它们的先后顺序），把拼接结果当作一个普通输入处理，测试集取得 0.71 acc。

### LSTM ✅测试集提升
改用 LSTM，自带时序属性，但当模型尺寸大时，虽然在训练集、验证集表现良好（acc 0.95+），但在测试集只有 0.73 左右。

怀疑模型过大过拟合，改小尺寸后验证集 0.83，测试集 0.71，较为缓解。

### 加权 CrossEntropyLoss
打印分类报告，发现大类主导，部分类别样本较少，召回率低，改为加权交叉熵损失函数，验证集 0.79，测试集 0.68。

调整权重，平滑惩罚力度，增大模型尺寸，提高训练集分割率，训练集 acc 0.95, f1 0.94，仍然测试集 0.73。

泛化能力弱。

### 局部卷积，感知增强
加入 局部卷积（Conv1D）前处理，验证集 0.88，测试集 0.71。

### Label Smoothing，缓解过拟合
加入 Label Smoothing，验证集 0.88，测试集 0.72。

### 提高拼帧 ✅测试集提升
这时候意识到输入的特征量太少了，把输入 n_concat=25，测试集达到了 0.76，通过 strong baseline。

### 全序列建模 ✅测试集提升
如果将整条序列直接暴力投给 LSTM，也许会有更好的结果，Public Score 0.80911，Private Score 0.80925。

试着在 LSTM 前加了 NormLayer 和 dropout，但并没有提升。

### 拼帧增大+全序列建模 ✅测试集提升
将拼好帧（25）丢给 LSTM，测试集达到 Public Score 0.82362, Private Score 0.82282。

试着把拼帧数量提高到 61，并无长进，测试集 0.822。

### 总结

拼帧可以显著提升局部上下文建模能力。

LSTM 需要整段输入+合理容量才能完全发挥效果。

权重/正则/结构调节可以缓解过拟合，但似乎测试集是另一个domain，提升有限。

全序列建模 + 拼帧 + LSTM  >  全序列建模 LSTM  >  拼帧 + LSTM（短序列输入）

可能：全序列建模可以建模跨语音段的上下文，如语速、韵律等，加上拼帧更好局部感受野扩展。

未来尝试：Conv1d 对拼帧后特征做降维 & More ...

> LSTM 中冗余数据可能有几个原因会让性能更好（这不是一定的）：
1. 虽然 LSTM 能捕捉时序依赖，但是需要通过学习一个 hidden state 来 capture，在数据输入中直接加入这些信息可以降低训练难度；
2. LSTM 只是缓解了梯度消失/爆炸的问题，帧堆叠可以提供更多的梯度反向传播路径，类似 LSTM 创建了梯度捷径，使得训练更稳定；
3. 从信息论角度，冗余信息可以形成一种“过完备”表示，可以增加对于噪声的容忍度，同时也和神经网络‘过参数化“的性质更好地 match
4. 从优化算法角度，冗余信息可以使得 landscape 更加 smooth，使得 SGD 这种随机优化方法效果更好
但是继续增加冗余度，会使得信息冗余带来的边际效益递减（可以从互信息角度解释），而且输入维度过高可能超过了 LSTM 的capacity，还可能会使得算法过拟合（可能是堆叠导致泛化能力弱的原因之一），噪声累积也可能是原因之一

## Reference

[1] colah, Understanding LSTM Networks, August 27, 2015 https://colah.github.io/posts/2015-08-Understanding-LSTMs/