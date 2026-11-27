---
title: 无偏权重导出与 served-copy 训练
tags:
  - 方法
created: 2026-11-02
modified: 2026-11-23
---

**一句话**：把高精度 master 无偏导出到低精度服务副本，并在真正生成数据的副本上计算训练 forward。

## 为什么需要

低 learning rate 的单步更新常小于 bf16 半格距，round-to-nearest 会连续多步丢掉它们。optimizer 以为参数已移动，sampler 却仍用旧 policy；下游 trainer/sampler ratio 因共享同一低精度副本而看不见错位。

## 怎么做

方法包含两个不可分割的动作：

1. export 使用 stochastic rounding，使每步 $\mathbb E[q_t]=\theta_t$；或使用 error feedback：

$$
q_t=\operatorname{round}(\theta_t+r_{t-1}),\qquad
r_t=\theta_t+r_{t-1}-q_t.
$$

2. sampler 与 trainer forward 都使用同一个 served copy $q_t$，gradient 通过 straight-through estimator 更新 fp32 master $\theta_t$。

上线诊断应同时记录 changed served weights fraction 和 master-versus-served policy log-ratio，不能只记录 trainer-versus-sampler ratio。

## 适用边界

weight unbiasedness 不推出 policy unbiasedness。Error feedback 需一份 fp32 residual；stochastic rounding 需可靠 RNG。方法增加每步变更权重数量，会降低 sparse delta sync 压缩率。若任务需要慢更新的正则效果，应显式对 master 做 EMA 后再无偏导出。

## 出处

- [[../40 每日报告/_历史单篇笔记/2026-11-02/bf16 Rounding Silently Discards Most RL Policy Updates - A Hidden Training-Inference Mismatch and an Unbiased One-Line Fix]]：第 4–5 节、Figure 2–4、Table 1

## 相关

- [[按来源分解梯度与有效更新检查]]
- [[梯度更新方向的信用分配]]
