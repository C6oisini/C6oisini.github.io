---
title: 自注意力 Self-Attention
tags:
  - 方法
created: 2026-10-23
modified: 2026-11-23
---

**一句话**：让序列里每个位置直接「看」所有其他位置，用相似度加权求和，一次矩阵乘法完成，没有循环。

## 核心公式

```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) V
```

| 符号  | 含义          | 常见形状             |
| --- | ----------- | ---------------- |
| Q   | 查询（当前要算的位置） | `[B, h, n, d_k]` |
| K   | 键（被查询的候选）   | `[B, h, n, d_k]` |
| V   | 值（实际被加权的内容） | `[B, h, n, d_v]` |

**为什么除以 √d_k**：`Q·K` 是 `d_k` 个独立同分布项求和，方差随 `d_k` 线性增长。不缩放的话 softmax 会进入饱和区，梯度趋近 0。

## 多头实现（伪代码）

```python
Q, K, V = qkv_proj(x).chunk(3, dim=-1)            # [B, n, d_model]
Q, K, V = [t.view(B, n, h, d_k).transpose(1, 2) for t in (Q, K, V)]

scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)   # [B, h, n, n]
scores = scores.masked_fill(mask, -inf)               # mask 要分情况，见下
attn   = torch.softmax(scores, dim=-1)
attn   = dropout(attn, p=0.1)
out    = (attn @ V).transpose(1, 2).reshape(B, n, d_model)
out    = out_proj(out)
```

## 踩坑清单

- [ ] **padding mask 和 causal mask 别混**：前者屏蔽 pad token（形状 `[B, n]`），后者屏蔽未来（`[n, n]` 上三角）。两个都要 `masked_fill`
- [ ] **mask 值用 `-inf` 还是 `-1e9`**：`-inf` 干净，但 fp16 下容易出 NaN，混合精度常改用 `torch.finfo(dtype).min`
- [ ] **dropout 加在 attention weights 上**（`attn` 之后），不要加在 softmax 之前
- [ ] **`view` 前要 `contiguous()`**，否则 reshape 报错或静默错位
- [ ] **`d_k` 别压太小**，为了省显存把每头维度压到 16 会掉点

## 复杂度

- 时间 `O(n²·d)`，空间 `O(n²)`（attention matrix）
- n ≤ 512 完全够用；n > 4096 就该考虑 FlashAttention / 稀疏 / 线性注意力

## 相关

- 首次提出：[[示例-Attention Is All You Need]]
- 变体：FlashAttention（IO 优化）、Multi-Query / Grouped-Query Attention（省 KV cache）
- 主题：[[Transformer 架构]]
