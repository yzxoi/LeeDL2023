# HW1 Regression 病例预测（回归）

> 2025.5.9 ~ 2025.5.11

## 学习目标

1. 用深度神经网络(DNN)拟合一个回归模型
2. 掌握DNN训练基本技巧

## TODO

[x] 特征选择 sklearn

[x] 优化器

[x] L2正则

[x] 超参 optuna

[x] tensorboard展示训练过程

### 特征选择 [1] [2]

#### Strategy 1. Filter: 对各个特征进行评分

- **Pearson相关系数**：线性相关性 
- **卡方验证**：仅适用于正数。
- **互信息和最大信息系数**：KL距离表述 $MI(x_i,y)=KL(P(x_i,y)||P(x_i)P(y))$，但不属于度量方式，无法归一化，且无法方便计算连续量。**最大信息系数**克服了这两个问题（先寻找一种最优的离散化方式，然后把互信息取值转换成一种度量方式，取值区间在 [0,1] ）
- **距离相关系数**：基于双中心化距离矩阵的内积 [3]。平方效率。
- **方差选择法**：选择特征方差大于阈值的特征

```python
SelectKBest(lambda X, Y: array(map(lambda x:pearsonr(x, Y), X.T)).T, k=2).fit_transform(X, y)
SelectKBest(chi2, k=2).fit_transform(X, y)
from minepy import MINE
SelectKBest(lambda X, Y: array(map(lambda x:mic(x, Y), X.T)).T, k=2).fit_transform(X, y)
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
print(sel.fit_transform(X))
```

#### Strategy 2. Wrapper: 根据目标函数筛选

对每个待选的特征子集，在训练集上训练一遍模型，然后在测试集上根据误差大小选择出特征子集。

需要先选定特定算法，通常选用普遍效果较好的算法， 例如Random Forest， SVM， kNN等等。

> 随着学习器（评估器）的改变，最佳特征组合可能会改变

- **前向搜索**：每次增量地从剩余未选中的特征选出一个加入特征集中，待达到阈值或者 n 时，从所有的 F 中选出错误率最小的。平方效率。
- **后向搜索**：默认选取所有特征，每次删除一个特征，并评价，直到达到阈值或者为空，然后选择最佳的 F 。平方效率。
- **递归特征消除法**：使用基模型来进行多轮训练，每轮训练后通过学习器返回的 coef_ 或者feature_importances_ 消除若干权重较低的特征，再基于新的特征集进行下一轮训练。

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

#递归特征消除法，返回特征选择后的数据
#参数estimator为基模型
#参数n_features_to_select为选择的特征个数
RFE(estimator=LogisticRegression(), n_features_to_select=2).fit_transform(iris.data, iris.target)
```

#### Strategy 3. Embedded: 用模型训练得出各特征权值系数

- **基于惩罚项的特征选择法**：L1正则方法具有稀疏解的特性，可以压缩一些系数恰好为0。L1没有选到的特征不代表不重要（两个具有高相关性的特征可能只保留了一个），如果要确定哪个特征重要应再通过L2正则方法交叉检验（在不同fold中稳定出现且系数很大）。

- **基于学习模型的特征排序**：直接使用你要用的机器学习算法，针对每个单独的特征和响应变量建立预测模型。假如某个特征和响应变量之间的关系是非线性的，可以用基于树的方法（决策树、随机森林）、或者扩展的线性模型等。但要注意过拟合问题，因此树的深度最好不要太大，再就是运用交叉验证。通过这种训练对特征进行打分获得相关性后再训练最终模型。

```python
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

#带L1惩罚项的逻辑回归作为基模型的特征选择   
SelectFromModel(LogisticRegression(penalty="l1", C=0.1)).fit_transform(iris.data, iris.target)

#带L1和L2惩罚项的逻辑回归作为基模型的特征选择   
#参数threshold为权值系数之差的阈值，C（逆正则强度）越小，惩罚越强，导致更多系数被压到零，保留特征越少。
SelectFromModel(LR(threshold=0.5, C=0.1)).fit_transform(iris.data, iris.target)

# 在波士顿房价数据集上使用sklearn的随机森林回归给出一个单变量选择的例子：
from sklearn.cross_validation import cross_val_score, ShuffleSplit
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor

#加载波士顿房价作为数据集
boston = load_boston()
X = boston["data"]
Y = boston["target"]
names = boston["feature_names"]

#n_estimators为森林中树木数量，max_depth树的最大深度
rf = RandomForestRegressor(n_estimators=20, max_depth=4)
scores = []
for i in range(X.shape[1]):
    #每次选择一个特征，进行交叉验证，训练集和测试集为7:3的比例进行分配，
    #ShuffleSplit()函数用于随机抽样（数据集总数，迭代次数，test所占比例）
    score = cross_val_score(rf, X[:, i:i+1], Y, scoring="r2",
                               cv=ShuffleSplit(len(X), 3, .3))
    scores.append((round(np.mean(score), 3), names[i]))

#打印出各个特征所对应的得分
print(sorted(scores, reverse=True))
```

### 优化器 [4] [5]

- **SGD（Stochastic Gradient Descent 随机梯度下降）**：单条数据（顺序随机）就可对参数进行一次更新。每个epoch参数更新M（样本数）次。更新速度快。每次训练少量数据，抽样偏差导致的参数收敛过程中**震荡**，当然 SGD 的震荡可能会跳到更好的局部极小值处。
- **BGD（Batch Gradient Descent 批量梯度下降）**：每次将所有样本的梯度求和，然后根据梯度和对参数进行更新，每个epoch参数更新1次。单次更新数据量大，梯度更新平滑。参数更新速度慢，内存消耗大，因为梯度更新平滑随机性差，**容易陷入局部最优解**。
- **MBGD（Mini-batch gradient descent  小批量梯度下降）**：每一次利用一小批样本，即 n 个样本进行计算，本质上就是在每个batch内部使用BGD策略，在batch外部使用SGD策略。没有考虑数据集的**稀疏度**和模型的训练时间对参数更新的影响。**不能保证很好的收敛性**（做法:学习率 大->小，两次迭代之间的变化低于某个阈值后，就减小 learning rate。此外希望对出现频率低的数据加大 lr），**容易停留在鞍点**（周围error相同）。
- **Momentum（动量梯度下降）**：$v_t=\gamma v_{t-1}+\eta\nabla_{\theta}J(\theta), \theta=\theta-\alpha v_t$ 引入动量，累计速度，减少震荡，使梯度更新平滑。有助于模型逃脱平稳区不易陷入局部极小值（**一阶指数平滑**）。每个批次的数据含有**抽样误差**，导致**梯度更新的方向波动较大**。超参建议值：$\gamma=0.9$ 左右。
- **NAG（Nesterov Accelerated Gradient 牛顿动量梯度下降）**：$v_t=\gamma v_{t-1}+\eta\nabla_{\theta}J(\theta_{t-1}-\alpha\gamma v_{t-1}), \theta_t=\theta_{t-1}-\alpha v_t$ 根据此次梯度和上次梯度的差值对Momentum算法得到的梯度进行修正，若差值为正，证明梯度增加，有理由相信下一个梯度会继续变大；相反两次梯度的差值为负，有理由相信下一个梯度会继续变小（**二阶指数平滑**）。**让算法提前看到前方的地形梯度**。缺点：不能根据参数的重要性对不同参数进行不同程度的更新。[6]
- **AdaGrad（Adaptive gradient algorithm 自适学习率应梯度下降）**：$s_t=s_{t-1}+(\nabla_{\theta}J(\theta))^2, \theta_t=\theta_{t-1}-\frac{\eta}{\sqrt{s_t+\epsilon}}\cdot \nabla_{\theta}J(\theta)$ 学习率逐渐下降，依据各参数变化大小调整学习率。每个参数的学习率反比于其历史梯度平方和的平方根，对出现频率较低的参数采用较大 $\alpha$ 更新。适合处理**稀疏数据**。$\eta$ 设置过大的话，会使分母过于敏感，对梯度的调节太大；中后期，分母上梯度平方的累加太大，易使训练**提前结束**（依赖于全局学习率）。超参建议值：$\eta =0.01$。
- **RMSprop（root mean square prop）**：$E[g^2]_t=\rho E[g^2]_{t-1}+(1-\rho)g_t^2,\theta_{t+1}=\theta_t-\frac{\eta}{\sqrt{E[g^2]_t+\epsilon}}\cdot g_t$ 对AdaGrad算法的改进，对历史的梯度信息使用decaying average的方式进行累计（用平方梯度的移动均值替代平方梯度的总和，一阶指数平滑处理）。通常与动量一起使用。但依然**依赖于全局学习率**。Pytorch中，`centered=True` 对梯度通过估计方差来进行归一化。[7] [8]
- **AdaDelta**：解决 RMSprop 中 $\eta$ 超参问题，维护 $\Delta x_t$ 存储模型本身中参数变化二阶导数的泄露平均值，$g_t$ 维护梯度二阶导数的泄露平均值。$E[\Delta x^2]_{t-1}=\rho E[\Delta x^2]_{t-2}+(1-\rho)\Delta x_{t-2}^2, \Delta x_t = \frac{\sqrt{E[\Delta x^2]_{t-1}+\epsilon}}{\sqrt{E[g^2]_t+\epsilon}},\Delta \theta_t=-\Delta x_t \cdot g_t$。由于初始阶段 $E[\Delta x^2]_{t-1}, E[g^2]_t$ 较小，一般在 $RMS$ 各除以 $1-\rho^t$ 来放大。训练后期，反复在局部最小值附近抖动。
- **Adam**：结合动量和自适应学习率，既能适应稀疏梯度又能缓解梯度震荡问题。使用指数加权移动平均值来估算梯度的动量和二次矩。$v_t=\beta_1 v_{t-1}+(1-\beta_1)g_t,s_t=\beta_2s_{t-1}+(1-\beta_2)g_t^2,\Delta \theta_t=-\frac{\eta \hat{v_t}}{\sqrt{\hat{s_t}}+\epsilon}$，$\hat{v_t}=\frac{v_t}{1-\beta_1^t}, \hat{s_t}=\frac{s_t}{1-\beta_2^t}$。与 RMSProp 区别即为 $\epsilon$  的处理，此方法在实践中效果略好。超参建议值：$\beta_1=0.9,\beta_2=0.999,\epsilon=1e-8$。
- **Yogi**：Adam算法即使在凸环境下，（重写Adam更新：$s_t=s_{t-1}+(1-\beta_2)(g_t^2-s_{t-1})$），当 $s_t$ 的二次矩估计值爆炸时，它可能无法收敛，故有改进版本：$s_t=s_{t-1}+(1-\beta_2) \text{sgn}(g_t^2-s_{t-1})$。现在更新的规模不再取决于偏差的量。
- **AdamW**：引入权重衰减，$\Delta \theta_t=-\frac{\eta \hat{v_t}}{\sqrt{\hat{s_t}}+\epsilon} -\lambda \theta_{t-1}$。实验结果表明 AdamW 比 Adam（利用动量缩小与 SGD 的差距）有更好的泛化性能，最优超参数的范围更广。
- **LARS**：$\lambda^l=\eta\cdot\frac{||w_t^l||_2^2}{||g_t^l||_2^2+\beta\cdot ||w_t^l||_2^2},v_{t+1}=\mu\cdot v_t+\alpha\cdot\lambda^l\cdot(g_t+\beta w_t),w_{t+1}=w_t-v_{t+1}$。批处理大小开始增长，但又会导致训练变得不稳定。LARS 是 SGD 的有动量扩展，可以适应每层的学习率。基于「信任」参数η<1 和该层梯度的反范数来重新调整每层的学习率。

```python
torch.optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0.99, centered=True)
torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,decoupled_weight_decay=False) #decoupled_weight_decay=True 则变为AdamW，忽略weight decay
```

tricks：

- 增加model随机性：shuffling, dropout, gradient noise
- Warm-up
- curriculum learning [Bengio, et al., ICML'09]: 先在容易的数据集上训练，再在困难的数据上训练
- Fine-tuning
- Normalization
- regulariza

## Practice

用了 k-cross，val 大概能收到 0.78 左右。

调参过程发现 lr 对结果的影响蛮大的，于是加了个周期性余弦退火，效果挺不错。

值得一提的是小模型参量少，选过多的特征可能不太好，另外记得一定要抠掉 id 等无关字段。

细节：k-fold 细节还是挺多的，每一折应独立取 kfeat，同时保存模型时也要同时保存该 kfeat 下的 test 数据便于后续 pred。

submission10.csv（自动选取 30 特征）：Private Score: 0.86127, Public Score: 0.87944

submission9.csv（手工选取 8 特征）：Private Score: 0.84726, Public Score: 0.80125

## Reference

[1] Scikit learn 1.13. Feature selection: https://scikit-learn.org/stable/modules/feature_selection.html

[2]知乎 [孙佳伟](https://www.zhihu.com/people/sun-jia-wei-34-83)【机器学习】特征选择(Feature Selection)方法汇总 https://zhuanlan.zhihu.com/p/74198735

[3] Github [josef-pkt](https://gist.github.com/josef-pkt)/try_distance_corr.py https://gist.github.com/josef-pkt/2938402)

[4] 知乎 [飞狗](https://www.zhihu.com/people/FlyingDogHuang) 优化器-optimizer 汇总 https://zhuanlan.zhihu.com/p/350043290

[5] DIVE INTO DEEP LEARNING 11. 优化算法 https://zh.d2l.ai/chapter_optimization/index.html

[6] Github WarBean/Momentum https://github.com/WarBean/zhihuzhuanlan/blob/master/Momentum_Nesterov.ipynb

[7] Geoff Hinton, Neural Networks for Machine Learning, Lecture 6a, Overview of mini-batch gradient descent https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf

[8] RMSprop, PyTorch https://docs.pytorch.org/docs/stable/generated/torch.optim.RMSprop.html

[9] Adam, PyTorch https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html