# 3. 基于模型的动态规划方法

# 3.1 基于模型的动态规划方法理论

强化学习可以归结为序贯决策问题。即找到一个决策序列，使得目标函数最优。

![](/assets/srqcqhxx_3_1.png)

强化学习可以用动态规划的思想求解，要使用动态规划，需要满足两个条件：

1. 整个优化问题可以分解为多个子优化问题；
2. 子优化问题可以被存储和重复利用。

强化学习中动态规划的核心是寻找最优值函数。对于状态值函数的迭代公式：


$$
v_{\pi}(s)=\sum_{a\in A}\pi(a|s)(R_s^a+\gamma \sum_{s' \in S}P_{ss'}^a v_\pi (s'))
$$


在模型已知的情况下，方程中的$$P_{ss'}^a$$，$$\gamma$$，$$R_s^a$$都是已知数，$$\pi(a|s)$$是要评估的策略，所以也是一致的。由此可见，方程中唯一的未知数是值函数，所以上面方程是关于值函数的线性方程组，其未知数的个数是状态的总数，用$$|S|$$表示。此处，可以用**高斯-赛德尔**方法求解：

#### 
$$
v_{k+1}(s) = \sum_{a \in A}\pi(a|s)(R_s^a+\gamma\sum_{s'\in S}P^a_{ss'}v_k(s'))
$$
策略评估算法

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

   2. 更新规则

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

















