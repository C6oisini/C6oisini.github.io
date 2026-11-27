---
title: Item-space On-Policy Distillation
tags:
  - 方法
created: 2026-11-02
modified: 2026-11-23
---

**一句话**：当学生和教师的 token 接口不同，用共享候选身份对齐教师对学生当前选择轨迹的条件反馈。

## 为什么需要

部署系统可能只输出一组 action scores，教师却逐步生成动作序列。直接做 token KL 要求共享词表，做 sequence SFT 又会把学生训练成与部署不同的生成器。像两位裁判用不同编号体系，应该先对齐运动员本人，而不是比较编号字符。

## 怎么做

先把学生分数排序成当前 top-$K$ 轨迹。第 $t$ 步删除已选 item，在相同剩余集合 $R_t$ 上分别归一化 student scores 与 teacher conditional scores：

$$
p_{\theta,t}(i)=\frac{e^{s_i}}{\sum_{j\in R_t}e^{s_j}},\qquad
p_{T,t}(i)=\frac{a_{T,t}(i)}{\sum_{j\in R_t}a_{T,t}(j)}.
$$

- $s_i$ 是学生部署时使用的 item score；$a_{T,t}(i)$ 是教师看过 student prefix 后对 item $i$ 的条件分数。
- 两个分布长度均为 $|R_t|$，索引按 item identity 对齐，不按 token ID 对齐。

再最小化各步 reverse KL，并与覆盖全候选的离线排序损失结合：

$$
L=L_{\mathrm{offline}}+\lambda\frac{1}{|T|}\sum_{t\in T}D_{\mathrm{KL}}(p_{\theta,t}\Vert p_{T,t}).
$$

## 适用边界

需要稳定的一一 item mapping，且 student score 至少能近似表示逐步条件偏好。候选装不进同一上下文、动作动态出现、已选集合强烈改变其余动作价值时，应加入 set-conditioned score，而不是机械重复归一化。教师成本随 $K$ 和训练步数增长。

## 出处

- [[../40 每日报告/_历史单篇笔记/2026-11-02/DecisionRank - On-Policy Distillation for Decision-Model Rerankers]]：第 4 节、Figure 2、Table 2–3

## 相关

- [[分段教师接管与局部蒸馏]]
- [[按来源分解梯度与有效更新检查]]
