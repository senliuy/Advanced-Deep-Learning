# 3. 基于模型的动态规划方法

## 3.1 基于模型的动态规划方法理论

强化学习可以归结为序贯决策问题。即找到一个决策序列，使得目标函数最优。

![](/assets/srqcqhxx_3_1.png)

强化学习可以用动态规划的思想求解，要使用动态规划，需要满足两个条件：

1. 整个优化问题可以分解为多个子优化问题；
2. 子优化问题可以被存储和重复利用。

强化学习中动态规划的核心是寻找最优值函数。对于状态值函数的迭代公式：


$$
v_{\pi}(s)=\sum_{a\in A}\pi(a|s)(R_s^a+\gamma \sum_{s' \in S}P_{ss'}^a v_\pi (s'))
$$


在模型已知的情况下，方程中的$$P_{ss'}^a$$，$$\gamma$$，$$R_s^a$$都是已知数，$$\pi(a|s)$$是要评估的策略，所以也是已知的，$$x = y$$表示后继状态的值函数。由此可见，方程中唯一的未知数是值函数，所以上面方程是关于值函数的线性方程组，其未知数的个数是状态的总数，用$$|S|$$表示。此处，可以用**高斯-赛德尔**方法求解：

#### 


$$
v_{k+1}(s) = \sum_{a \in A}\pi(a|s)(R_s^a+\gamma\sum_{s'\in S}P^a_{ss'}v_k(s'))
$$


#### 策略评估算法

```
输入：需要评估的策略pi和状态转移概率p,回报函数r,折扣因子gamma
初始化值函数：V(s)=0
Repeat k=0,1,...
    for every s do
        execute Gaussian-Seidel method
    end for
Until u[k+1] = u[k]
```

网格世界的例子，需要补充两点

1. 衰减系数$$\gamma = 1$$
2. 对于边界情况，用当前位置的值补充遍历无方向的值

更详细的可以参考[知乎讨论](https://zhuanlan.zhihu.com/p/28084990)

两个重要代码：

1. 边界处理

```py
# 根据当前状态和采取的行为计算下一个状态id以及得到的即时奖励
def nextState(s, a):
  next_state = s
  if (s%4 == 0 and a == "w") or (s<4 and a == "n") or \
     ((s+1)%4 == 0 and a == "e") or (s > 11 and a == "s"):
    pass
  else:
    ds = ds_actions[a]
    next_state = s + ds
  return next_state
```

1. 更新规则

```py
# update the value of state s
def updateValue(s):
  sucessors = getSuccessors(s)
  newValue = 0  # values[s]
  num = 4       # len(successors)
  reward = rewardOf(s)
  for next_state in sucessors:
    newValue += 1.00/num * (reward + gamma * values[next_state])
  return newValue
```

得到更新后的值函数后，贪心算法是一种最常见的策略改善算法，即$$\pi_{l+1}(s) \in argmax_a q^{\pi_l}(s,a)$$

至此，已经可以得到策略改善算法

#### 策略改善算法

```
输入：需要评估的策略pi和状态转移概率p,回报函数r,折扣因子gamma
初始化值函数：V(s)=0
Repeat k=0,1,...
    for every s do
        策略评估：高斯赛德尔算法
        策略改善：贪心算法
    end for
Until u[k+1] = u[k]
```

不一定要等到策略值函数收敛之后再进行策略改善。如果我们在评估一次之后就进行策略改善，则称之为**值函数迭代算法。**

## 3.2 动态规划中的数学基础讲解

### 3.2.1 线性方程组的迭代解法

状态值函数的计算公式


$$
v_{\pi}(s)=\sum_{a\in A}\pi(a|s)(R_s^a+\gamma \sum_{s' \in S}P_{ss'}^a v_\pi (s'))
$$


是一个线性方程组，求解线性方程组有直接法和简介迭代法。策略评估中采用的是迭代解法。

#### 什么是迭代解法

一般的线性方程组可以表示为：


$$
AX=b
$$


所谓迭代解法就是根据上式设计一个迭代公式，代入初始值$$X^{(0)}$$，将其代入迭代公式得到$$X^{(1)}$$，再代入$$X^{(1)}$$得到$$X^{(2)}$$，如此循环直到收敛到$$X$$

求解迭代公式有两种解法：

（1）雅克比迭代法

（2）高斯-赛德尔迭代法

### 3.2.2 压缩映射证明策略评估的收敛性

不动点和压缩映射常用来解决代数方程，微分方程，积分方程等，为方程解的存在性，唯一性和讨论迭代收敛性证明提供有力的工具。

定义：设X是度量空间，其度量用$$\rho$$表示。映射$$T:X\rightarrow X$$，若存在$$a$$，$$0 \leq a < 1$$使得$$\rho(Tx, Ty) \leq \rho(x,y), \forall x,y \in X$$，则称T是X上的一个**压缩映射**。

若存在$$x_0 \in X$$使得$$Tx_0 = x_0$$，则称$$x_0$$是$$T$$的**不动点**。

#### 定理1：完备度量空间上的压缩映射具有唯一的不动点。

定理1是说：从度量空间任何一点出发，只要满足压缩映射，压缩映射的序列必定会收敛到唯一的不动点。所以可以通过证明压缩映射来评估策略的收敛性。

## 3.3 基于gym的变成实例

下面给出重要函数的代码片段

#### 策略迭代方法

```py
def policy_evaluate(self, grid_mdp):
    for i in range(1000):
        delta = 0.0
        for state in grid_mdp.states:
            if state in grid_mdp.terminal_states: continue
            action = self.pi[state]
            t, s, r = grid_mdp.transform(state, action)
            new_r = r + grid_mdp.gamma * self.v[s]
            delta += abs(self.v[state] - new_v)
            self.v[state] = new_v
        if delta < 1e-6:
            break
```

#### 策略改善方法

```
def policy_improve(self, grid_mdp):
    for state in grid_mdp.states:
        if state in grid_mdp.terminal_states: continue
        a1 = grid_mdp.actions[0]
        t, s, r = grid_mdp.transform(state, a1)
        v1 = r + grid_mdp.gamma * v[s]
        for action in grid_mdp.actions:
            t, s, r = grid_mdp.transform(state, action)
                if v1 < r + grid_mdp.gamma * v[s]:
                    a1 = action
                    v1 = r + grid_mdp.gamma * v[s]
        self.pi[state] = a1
```



