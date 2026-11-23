"""HF 参考实现 (从 modeling_glm5_next.py 逐行移植) — 测试的黄金参考.

只用于测试对拍; 训练代码不 import 本文件.
"""

import torch
import torch.nn.functional as F


def hf_l2norm(x, dim=-1, eps=1e-6):
    """HF l2norm (MG:459): 刻意 sqrt + / (非 F.normalize)."""
    inv_norm = torch.sqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x / inv_norm


def hf_recurrent_kda(query, key, value, g, beta, initial_state=None):
    """HF recurrent_kimi_delta_attention (MG:462-513) — 朴素逐步 delta rule.

    输入 [B,S,H,Dk] (q/k/v), g [B,S,H,Dk], beta [B,S,H].
    输出 (core_attn_out [B,S,H,Dv], final_state [B,H,Dk,Dv]), fp32 内部.
    """
    initial_dtype = query.dtype
    query, key, value, g, beta = [x.to(torch.float32) for x in (query, key, value, g, beta)]
    query = hf_l2norm(query, dim=-1, eps=1e-6)
    key = hf_l2norm(key, dim=-1, eps=1e-6)

    batch_size, sequence_length, num_heads, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(
        batch_size, sequence_length, num_heads, v_head_dim, dtype=value.dtype, device=value.device)
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim,
                    dtype=value.dtype, device=value.device)
        if initial_state is None else initial_state.to(value))

    for i in range(sequence_length):
        q_i = query[:, i]
        k_i = key[:, i]
        v_i = value[:, i]
        g_i = g[:, i][..., None].exp()
        b_i = beta[:, i][..., None]

        last_recurrent_state = last_recurrent_state * g_i
        kv_mem = (last_recurrent_state * k_i[..., None]).sum(dim=-2)
        delta = (v_i - kv_mem) * b_i
        last_recurrent_state = last_recurrent_state + k_i.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, i] = (last_recurrent_state * q_i.unsqueeze(-1)).sum(dim=-2)

    return core_attn_out.to(initial_dtype), last_recurrent_state


def hf_hyper_connection_gates(streams, fn, base, scale, n, eps, iters):
    """HF Glm5NextTextHyperConnection.forward (MG:277-295) 的门控部分.

    streams [b,s,n,h] (float); 返回 (pre, post, comb).
    """
    b, s, _, h = streams.shape
    flattened = streams.reshape(b, s, -1).float()
    flattened = flattened * torch.rsqrt(flattened.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
    w = torch.nn.functional.linear(flattened, fn.float())
    pre_w, post_w, comb_w = w.split([n, n, n * n], dim=-1)
    pre_b, post_b, comb_b = base.float().split([n, n, n * n], dim=0)
    pre_s, post_s, comb_s = scale.float().unbind(0)

    comb_w = comb_w.view(b, s, n, n)
    comb_b = comb_b.view(n, n)
    pre = torch.sigmoid(pre_w * pre_s + pre_b) + eps
    post = 2 * torch.sigmoid(post_w * post_s + post_b)
    comb = torch.softmax(comb_w * comb_s + comb_b, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def hf_rmsnorm(x, weight, eps):
    """HF Glm5NextTextRMSNorm: fp32 计算后回原 dtype."""
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x * weight.float()).to(dtype)
