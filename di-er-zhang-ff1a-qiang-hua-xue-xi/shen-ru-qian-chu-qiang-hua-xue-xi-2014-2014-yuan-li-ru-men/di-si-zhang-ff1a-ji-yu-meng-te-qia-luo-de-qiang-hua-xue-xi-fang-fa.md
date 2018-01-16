# 4.1 基于蒙特卡洛方法的理论

无模型的强化学习算法主要包括蒙特卡洛方法和时间差分法，如下图

![](/assets/srqcqhxx_4_1.png)

状态值函数和行为值函数的计算实际上是计算返回值的期望，动态规划的思想是利用模型计算该期望。在没有模型时，我们可以采用蒙特卡洛的方法计算该期望，即利用随机样本估计期望。**在计算值函数时，蒙特卡罗方法是利用经验平均代替随即变量的期望**。

**经验**：利用该策略做很多次实验，产生很多组数据。

**平均**：求均值。

1. 第一次访问蒙特卡罗方法：在计算s处的值函数时，只利用每次实验中第一次访问到状态s时的返回值。
2. 每次访问蒙特卡罗方法：在计算s处的值函数时，利用每次实验中所有访问到状态s时的返回值。

#### 获得足够多的经验是无模型强化学习的核心所在。

蒙特卡罗方法中必须采取一定的策略来保证每个状态都能被访问到。

###### 探索性初始化蒙特卡罗方法

```
[1] 初始化所有：
    s ∈ S, a ∈ A(s), Q(s,a)←Arbitrary, π(s)←Arbitrary, Returns(s, a)←Empty List
[2] Repeat:
        随机选择 S0 ∈ S, A0 ∈ A(S0), 从A0,S0开始策略V, 生成一个实验（episode）, 对每队在这个实验中出现的状态和动作s,a:
[3]         #策略评估
            G←s,a第一次出现后的回报
            将G附加于回报Returns(s,a)上
            Q(s,a)←average(Returns(s,a))对回报取均值
[4] 对实验中的每一个s:
        #策略改进
        π(s)←argmax_a Q(s,a)
```

探索性初始化在迭代每一幕时，初始状态时随机分配的。初始化需要精心设计，以保证每一个状态都有可能被访问到，即对任意状态s，a都满足：$$\pi(a|s) > 0$$，折中策略叫做温和的，一个典型的策略是$$\varepsilon-soft$$策略：


$$
\pi(a|s)=
\begin{cases}
1-\varepsilon+\frac{\varepsilon}{|A(s)|}&a=argmax_aQ_*(s,a)\\
\frac{\varepsilon}{|A(s)|}&otherwise
\end{cases}
$$


根据探索策略和评估策略是否是同一个策略，蒙特卡洛方法分成同策略（on-policy）和异策略（off-policy）：

1. 同策略：产生数据的策略与评估和要改善的策略是同一个策略。
2. 异策略：产生数据的策略\(μ\)与评估和要改善的策略\(π\)不是同一个策略。

基于异策略的目标策略（π）和行动策略（μ）的旋转要满足**覆盖性条件，**即行动策略产生的行为要覆盖或者包含目标策略产生的行为。

利用行为策略产生的数据评估目标策略需要利用**重要性采样**方法。

重要性采样来源于求期望


$$
E[f] = \int f(z)p(z)dz
$$


![](/assets/srqcqhxx_4_2.png)

上图中的p\(z\)是一个非常复杂的分布，无法通过解析的方法产生用于逼近期望的样本，这时，可以采用一个简单的分布。原来的分布可以变成


$$
E[f]= \int f(z)\frac{p(z)}{q(z)}q(z)dz \approx \frac{1}{N} \sum_n \frac{p(z^n)}{q(z^n)}f(z^n), z^n \sim q(z)
$$


因为上面的f\(z\)很难计算，所以可以通过大量实验来拟合期望。

定义重要性权重：$$\omega^n = p(z^n)/q(z^n)$$，普通的重要性采样求积分变成方程：


$$
E[f] = \frac{1}{N}\sum_n \omega^n f(z^n)
$$


该估计为无偏估计（估计的期望等于真实期望），但是方差为无穷大，一种减小方差的方法是采用加权重要性采样。


$$
E[f] \approx \sum^N_{n=1}\frac{\omega^n}{\sum^N_{m=1}\omega^m}f(z^n)
$$


回归到蒙特卡洛方法，行动策略μ用来产生样本，对应的轨迹是重要性采样中的q\[z\]，用来评估和改进的策略π对应的轨迹概率分布是p\[z\]。加权重要性采样值函数估计为


$$
V(s) = \frac{\sum_{t\in \mathcal{T}(s)\rho^(T(t))_t}G_t}{\sum_{t\in \mathcal{T}(s)\rho^(T(t))_t}}
$$


其中G\(t\)是从t到T\(t\)的返回值


$$
\rho^T_t = \prod_{k=t}^{T-1}\frac{\pi(A_k|S_k)}{\mu(A_k|S_k)}
$$


###### 蒙特卡洛方法伪代码

```
[1] 初始化所有：
    s ∈ S, a ∈ A(s), Q(s,a)←Arbitrary, C(s,a)←0, π(s)←相对于Q的贪婪策略
[2] Repeat forever
    利用软测率μ产生一次实验
    S[0], A[0], R[1], ..., S[T-1], A[T-1], R[T],S[T]
    G←0， W←0
[3] for t = T-1, T-2, ..., 0
    G←gamma*G+R[t-1]
    C(S[t], A[t]) ← C(S[t], A[t]) + W #策略评估
    G(S[t], A[t]) ← G(S[t], A[t]) + W/C(S[t], A[t])*(G-Q(S[t], A[t])) #重要性采样
    π(S[t])←argmax_a Q(s[t], a) #策略改善 #策略改善
    如果A[t] != π(s[t])则退出循环
    W←W/μ(A[t]|S[t])
```

**总结**：重点讲解了如何利用MC的方法估计值函数。与基于动态规划的方法相比，基于MC的方法只是在值函数估计上有所不同，在整个框架上则是相同的，即评估当前策略，在利用学到的值函数进行策略改善。本届需要重点理解on-policy和off-policy的概念，并学会利用重要性采样来评估目标函数的值函数。

---

## 4.2 统计学基础知识

当无法通过模型更新值函数时，可以通过经验平均代替值函数。使用经验平均就涉及到采样，如何保证采样样本的无偏估计是非常重要的，通常有两类方案。

1. 使用一个已知的概率p来采样，包括拒绝采样和重要性采样，但此方法只适用于低维情况
2. 第二种是马尔科夫蒙特卡洛方法（MCMC）采样

### MCMC

\[1\] 可以参考这篇文章：[https://www.jianshu.com/p/27829d842fbc](https://www.jianshu.com/p/27829d842fbc)

MCMC不需要提议分布，只需要一个随机样本点，下一个样本会由当前的随机样本点产生，如此循环源源不断地产生很多样本点，最终这些样本点服从目标分布。

MCMC背后的原理是：目标分步为**马氏链平稳分布**：

该目标分步存在一个转移概率矩阵P，满足$$\pi(j) = \sum_{i=0}^{\infty}\pi(i)P_{ij}$$; $$\pi$$是方程的唯一非负解。

当转移矩阵满足此条件时，从任意出事分步$$\pi_0$$出发，经过一段时间迭代，分步$$\pi_t$$都会收敛到目标分步$$\pi$$。给定一个初始状态$$x_0$$，那么会得到连续的转移序列$$x_0,x_1,...,x_n,x_{n+1},...$$。如果该马氏链在n时收敛到目标分步$$\pi$$，我们就得到了目标分步的样本$$x_n,x_{n+1},...$$。那么如何构造转移概率$$P$$呢？

定理：细致平稳条件

如果非周期马氏链的转移矩阵$$P$$和分布$$\pi(x)$$满足


$$
\pi(i)P_{ij} = \pi(j)P_{ji}; \forall i,j
$$


则$$\pi(x)$$是马氏链的平稳分布，上式叫做细致平稳条件。

假设我们已经有一个转移矩阵为$$Q$$的马氏链，显然，通常情况下：


$$
p(i)q(i,j) \neq p(j)q(j,i)
$$


也就是细致平稳条件不成立，所以$$p(x)$$不太可能是这个马氏链的平稳分布。我们可否对马氏链做一个改造，使得细致平稳条件成立呢？譬如，我们引入一个$$\alpha(i,j)$$, 我们希望：


$$
p(i)q(i,j)\alpha(i,j) =p(j)q(j,i)\alpha(j,i)
$$


取什么样的$$\alpha(i,j)$$以上等式能成立呢？最简单的，按照对称性，我们可以取：


$$
\alpha(i,j) = p(j)q(j,i), \alpha(j,i) = p(i)q(i,j)
$$


于是上式式就成立了。

于是我们把原来具有转移矩阵$$Q$$的一个很普通的马氏链，改造为了具有转移矩阵$$Q'$$ 的马氏链，而$$Q'$$恰好满足细致平稳条件，由此马氏链$$Q'$$的平稳分布就是$$p(x)$$。

在改造$$Q$$的过程中引入的$$\alpha(i,j)$$称为接受率，物理意义可以理解为在原来的马氏链上，从状态$$i$$以$$q(i,j)$$的概率转跳转到状态$$j$$的时候，我们以$$α(i,j)$$的概率接受这个转移，于是得到新的马氏链$$Q'$$的转移概率为$$q(i,j)α(i,j)$$。

![](/assets/srqcqhxx_4_3.png)

假设我们已经有一个转移矩阵$$Q$$\(对应元素为$$q(i,j)$$\), 把以上的过程整理一下，我们就得到了如下的用于采样概率分布$$p(x)$$的算法。

###### MCMC采样算法

```
[1] 初始化马氏链初始状态X[0] = x[0]
[2] 对t=1,2,3..., 循环进行下面采样
        第t个时刻马氏链状态为X[t] = x[t], 采样y \sim q(x|x[t])
        从均匀分布中采样 u \sim Uniform[0,1]
        if u < α(x[t],y) = p(y)q(x[t]|y): 接受转移x[t] → y, 即X[t+1] = y
        else 不接受转移，即X[t+1]=x[t]
```

为了提高接受率，使样本多样化，Metropolis-Hasting算法将$$α(i,j)$$改为


$$
\alpha(i,j) = min\{\frac{p(y)q(x_t|y)}{p(x_t)q(y|x_t)}, 1\}
$$


---

## 4.3 基于Python的编程实例

利用蒙特卡洛方法评估策略应该包括两个过程：模拟和平均。

1. 随机策略的样本产生：模拟

###### 蒙特卡洛样本采集

```py
def gen_randomi_sample(self, num):
    state_sample = []
    action_sample = []
    reward_sample = []
    for i in range(num):
        s_tmp = []
        a_tmp = []
        r_tmp = []
        s = self.states[int(random.random() * len(self.states))] #随机初始化每回合的初始状态
        t = False
        while False == t: #产生一个状态序列
            a = self.actions[int(random.random() * len(self.actions))]
            t, s1, r= self.transform(s, a)
            s_tmp.append(s)
            a_tmp.append(a)
            r_tmp.append(r)
            s = s1
        state_sample.append(s_tmp)
        action_sample.append(a_tmp)
        reward_sample.append(r_tmp)
    return state_sample, action_sample, reward_sample
```

  2. 得到值函数：平均

###### 蒙特卡洛评估

```py
def mc(gamma, state_sample, action_sample, reward_sample):
    vfunc = dict()
    nfunc = dict()
    for s in states
        vfunc[s] = 0.0
        nfunc[s] = 0.0
    for iter1 in range(len(state_sample)):
        G = 0.0
        for step in range(len(state_sample[iter1])-1, -1, -1):
            G *= gamma
            G += reward_sample[iter1][step]
        for step in range(len(state_sample[iter1])
            s = state_sample[iter1][step]
            vfunc[s] += G
            nfunc[s] += 1.0
            G -= reward_sample[iter1][step]
            G /= gamma
    for s in states:
        if nfunc[s] > 0.000001:
        vfunc[s] /= [s]
    return vfunc
```



