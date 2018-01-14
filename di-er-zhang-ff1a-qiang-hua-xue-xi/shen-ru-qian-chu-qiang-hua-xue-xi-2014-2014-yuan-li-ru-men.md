# 第一篇：强化学习基础

# 2. 马尔科夫决策过程

## 2.1 马尔科夫决策过程理论讲解

#### 马尔科夫性

**马尔科夫性**：系统的下一个状态$$s_{t+1}$$仅与当前状态$$s_t$$有关，而与之前的状态无关

定义：状态$$s_t$$是马尔科夫的，当且仅当$$P[s_{t+1}|s_{t}]=P[s_{t+1}|s_1,...,s_t]$$。

**马尔科夫随机过程**：随机变量序列中的每个状态都是马尔科夫的。

#### 马尔科夫过程

**马尔科夫过程**是一个二元组\(S, P\), 且满足：S是有限状态集合，P是状态转移概率。

**马尔科夫决策过程**：将动作（策略）和回报考虑在内的马尔科夫过程成为马尔科夫随机过程。

### 马尔科夫决策过程

马尔科夫决策过程由元组$$(S, A, P, R, \gamma)$$描述，其中:

* S为有限的状态集
* A为有限的动作集
* P为状态转移概率
* R为回报函数
* $$\gamma$$为折扣因子，用来计算累积回报

马尔科夫决策过程的的状态转移概率是包含动作的，即$$P^{a}_{ss'}=p[S_{t+1}=s'|S_t=s,A_t=a]$$。

强化学习的目标是给定一个马尔科夫决策过程，寻找**最优策略**。

最优策略：是指状态到动作的映射，策略通常用符号$$\pi$$表示，它是指给定状态$$s$$时，动作集上的一个分布，即


$$
\pi(a|s)=p[A_t=a|S_t=s]
$$


它的含义是：策略$$\pi$$在每个状态$$s$$指定一个动作概率。如果给出的策略$$\pi$$是确定性的，那么策略$$\pi$$在每个状态$$s$$指定一个确定的动作。

累积回报：


$$
G_t=R_{t+1}+\gamma R_{t+2}+...=\sum^{\infty}_{k=0}\gamma^kR_{t+k+1}
$$


（1）状态-值函数：累积回报$$G_1$$的期望

定义：当智能体采用策略pi时，累积回报服从一个分布，累积回报在状态s处的期望值定义为状态-值函数：


$$
v_{\pi}(s)=E_{\pi}[\sum^{\infty}_{k=0}\gamma^kR_{t+k+1}|S_t=s]
$$


相应的，状态-行为值函数定义为：


$$
q_{\pi}(s,a)=E_{\pi}[\sum^{\infty}_{k=0}\gamma^kR_{t+k+1}|S_t=s,A_t=a]
$$


状态值函数的贝尔曼方程


$$
v_{\pi}(s)=E_{\pi}[R_{t+1}+\gamma v_{\pi}(S_{t+1})|S_t=s]
$$


状态行为值函数的贝尔曼方程


$$
q_{\pi}(s,a)=E_{\pi}[R_{t+1}+\gamma q_{\pi}(S_{t+1}, A_{t+1})|S_t=s, A_t=a]
$$


状态值函数的计算方式


$$
v_{\pi}(s)=\sum_{a\in A}\pi(a|s)(R_s^a+\gamma \sum_{s' \in S}P_{ss'}^a v_\pi (s'))
$$


状态行为值函数


$$
q_\pi (s,a) = R_s^a+\gamma \sum_{s' \in S} P_{ss'}^a \sum_{a' \in A}\pi(a'|s')q_{\pi}(s',a')
$$


最优状态值函数$$v^*(s)$$为在所有策略中值最大的值函数，即$$v^*(s)=max_\pi v_\pi(s)$$。

最优状态-行为值函数$$q^*(s,a)$$为在所有策略中最大的状态-行为值函数，即$$q^*(s,a)=max_\pi q_\pi(s,a)$$

定义一个离散时间有限范围的折扣马尔科夫决策过程$$M=(S,A,P,r,\rho_0,\gamma,T)$$其中$$S$$为状态集，$$A$$为动作集，$$P: S\times A \times S \rightarrow R$$是转移概率， $$r:S\times A\rightarrow[-R_{max}, R_{max}]$$为立即回报函数，$$\rho_0: S \rightarrow R$$是初始状态分布，$$\gamma \in [0,1]$$是折扣因子，T为水平范围（其实是步数）。$$\tau$$为一个轨道序列，即$$\tau = (s_0,a_0,s_1,a_1,...)$$，累积回报为$$R = \sum_{t=0}^T \gamma^t r^t$$,强化学习的目标就是找到最优策略$$\pi$$，使得该策略下的累积回报期望最大，即$$max_{\pi}\int R(\tau)p_\pi (\tau)d\tau$$。

## 2.2 MDP中的概率学基础讲解

强化学习中常采用的随机策略。

（1）贪心策略


$$
\pi_*(a|s)=\begin{equation}
\begin{cases}
1&a=argmax_a\in Aq_*(s,a)\\
0&otherwise
\end{cases}
\end{equation}
$$


（2）$$\varepsilon$$-greedy策略


$$
\pi_*(a|s)=\begin{equation}
\begin{cases}
1-\varepsilon+\frac{\varepsilon}{|A(s)|}&a=argmax_a\in Aq_*(s,a)\\
\frac{\varepsilon}{|A(s)|}&otherwise
\end{cases}
\end{equation}
$$


（3）高斯策略


$$
\pi_\theta = \mu + \varepsilon, \varepsilon~N(0,\sigma^2)
$$


（4）玻尔兹曼分布


$$
\pi(a|s,\theta)=\frac{exp(Q(s,a,\theta))}{\sum_b exp(Q(s,b,\theta))}
$$


1. 基于模型的动态规划方法



