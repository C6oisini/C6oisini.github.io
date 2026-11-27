---
title: Attention Is All You Need
authors:
  - Vaswani
  - Shazeer
  - Parmar
  - Uszkoreit
  - Jones
  - Gomez
  - Kaiser
  - Polosukhin
year: 2017
venue: NeurIPS 2017
url: https://arxiv.org/abs/1706.03762
status: 已读
rating: 5
topics:
  - 序列建模
  - 机器翻译
methods:
  - 自注意力
tags:
  - 论文
created: 2026-10-23
updated: 2026-10-23
---

> [!abstract] 一句话总结
> 把 RNN/CNN 从序列建模里彻底删掉，只用自注意力堆出 Transformer，在 WMT14 英德上刷到 28.4 BLEU，训练成本还降了。

> [!warning] 这是示例笔记
> 用来演示模板怎么填。你读完第一篇真论文后可以直接删掉。

## 1. 研究问题

- 背景：当时的 SOTA 序列模型（RNN/LSTM/GRU）必须**按时间步串行**计算，无法在序列维度并行，长序列训练极慢
- 要解决的问题：能否在不牺牲质量的前提下，去掉循环结构，让序列建模完全并行？
- 为什么重要：并行度直接决定能不能吃下更多算力和更长上下文

## 2. 核心贡献

- [x] 提出 **Transformer**：首个完全基于注意力、不含循环和卷积的序列转换架构
- [x] **多头自注意力**：把表示拆成 h 个子空间分别做注意力再拼回
- [x] **正弦位置编码**：用固定函数注入位置信息，不引入额外参数
- [x] 训练效率：8 张 P100 训 12 小时达到英德 SOTA

## 3. 方法

- 整体：encoder–decoder 各 **6 层**，每层 `[多头注意力 → Add&Norm → FFN → Add&Norm]`
- 超参：`d_model=512`，`h=8`，`d_k=d_v=64`，FFN 隐层 `2048`，dropout `0.1`
- 注意力：`Attention(Q,K,V) = softmax(QKᵀ/√d_k)V`，缩放防止点积过大导致梯度消失
- 位置编码：`PE(pos,2i)=sin(pos/10000^(2i/d_model))`，偶数维 sin、奇数维 cos
- 训练：Adam（`β2=0.98`），warmup 4000 步后按步数平方根倒数衰减；label smoothing `0.1`
- 解码：beam search，beam=4，长度惩罚 α=0.6

详细机制见 [[自注意力 Self-Attention]]。

## 4. 实验

| 数据集 | 指标 | 本文 (big) | 此前最好（集成） | 提升 |
| --- | --- | --- | --- | --- |
| WMT14 EN-DE | BLEU | 28.4 | 26.36 (ConvS2S) | +2.0 |
| WMT14 EN-FR | BLEU | 41.8 | 41.29 (ConvS2S) | +0.5 |
| 训练 FLOPs | FLOPs | 2.3e19 | 1.4e20 (GNMT+RL) | 约 1/6 |

消融（Table 3）里值得看的三条：

- 单头注意力比 8 头差约 0.9 BLEU → **多头确实有用**
- 去掉位置编码掉 1.3 BLEU，但换成可学习的位置嵌入结果几乎一样
- `d_k` 调小反而变差，说明缩放不是随便加的

## 5. 关键结论

- 注意力单独就足以建模序列，循环不是必需的
- 任意两个位置之间的路径长度从 O(n) 降到 O(1)，这是并行和效果双赢的根源

## 6. 局限与我的质疑

- 作者承认的局限：自注意力是 **O(n²)** 复杂度，长序列直接爆显存
- 我发现的漏洞：
  - **位置编码的外推性没有实验**，训练长度之外的性能完全没提
  - 与 CNN 的对比只在翻译任务上做，其他模态没有验证
  - 正弦 vs 可学习编码差距在噪声范围内，「选正弦是为了外推」这个理由缺乏证据
- 实验是否支撑结论：**翻译任务上支撑得很充分**；但「Transformer 是通用架构」这个更大的 claim 在 2017 年还没有证据（要等 BERT / ViT）

## 7. 可复用点

- 想在自己的模型里替换 RNN → 直接抄 `MultiHeadAttention` 实现，注意 **mask 要分开处理**（padding mask 和 causal mask 别混成一个）
- 训练不收敛先查三件事：warmup 步数、`√d_k` 缩放、label smoothing
- 显存不够时优先降 `d_model` 而不是降层数，层数对效果更敏感

## 8. 原文摘录

> "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely."

> "The Transformer ... trains significantly faster than architectures based on recurrent or convolutional layers."

## 9. 相关

- 前置：Bahdanau Attention（第一个把注意力用在 seq2seq 上的工作）
- 后续：BERT、GPT 系列、ViT
- 同主题：[[Transformer 架构]]
