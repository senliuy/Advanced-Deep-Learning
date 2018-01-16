```
[3] for t = T-1, T-2, ..., 0
    G←gamma*G+R[t-1]
    C(S[t], A[t]) ← C(S[t], A[t]) + W #策略评估
    G(S[t], A[t]) ← G(S[t], A[t]) + W/C(S[t], A[t])*(G-Q(S[t], A[t])) #重要性采样
    π(S[t])←argmax_a Q(s[t], a) #策略改善 #策略改善
    如果A[t] != π(s[t])则退出循环
    W←W/μ(A[t]|S[t])
```



