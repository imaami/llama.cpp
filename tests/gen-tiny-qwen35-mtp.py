#!/usr/bin/env python3
# Generate a small seeded Qwen3.5 trunk plus one MTP head for CPU integration tests.
# Based on PrismML-Eng PR #116's tiny-Qwen fixture; uses a self-contained byte vocab.
# Usage: python3 tests/gen-tiny-qwen35-mtp.py <output.gguf>

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gguf-py"))
from gguf import GGUFWriter, GGUFValueType  # noqa: E402

N_LAYER      = 4    # layers 0,1,2 recurrent (GDN), layer 3 full attention
N_EMBD       = 64
N_FF         = 128
N_HEAD       = 2
N_HEAD_KV    = 1
N_CTX        = 512
D_CONV       = 4
D_STATE      = 16   # GDN head dim (S_v)
N_V_HEADS    = 4    # ssm.time_step_rank
N_K_HEADS    = 2    # ssm.group_count
D_INNER      = D_STATE * N_V_HEADS          # 64
KEY_DIM      = D_STATE * N_K_HEADS          # 32
VALUE_DIM    = D_STATE * N_V_HEADS          # 64
CONV_DIM     = 2 * KEY_DIM + VALUE_DIM      # 128
HEAD_K_DIM   = N_EMBD // N_HEAD             # 32


def main() -> None:
    out_path = sys.argv[1] if len(sys.argv) > 1 else "tiny-qwen35-mtp.gguf"

    rng = np.random.default_rng(1234)

    def rand(*shape: int) -> np.ndarray:
        return rng.normal(0.0, 0.02, size=shape).astype(np.float32)

    w = GGUFWriter(out_path, "qwen35")
    w.add_name("tiny-random-qwen35-mtp")
    w.add_file_type(0)  # all f32

    w.add_block_count(N_LAYER + 1)
    w.add_uint32("qwen35.nextn_predict_layers", 1)
    w.add_context_length(N_CTX)
    w.add_embedding_length(N_EMBD)
    w.add_feed_forward_length(N_FF)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_HEAD_KV)
    w.add_layer_norm_rms_eps(1e-6)
    w.add_rope_freq_base(10000.0)
    # IMROPE sections must sum to head_dim/2
    w.add_rope_dimension_sections([HEAD_K_DIM // 4, HEAD_K_DIM // 8, HEAD_K_DIM // 8, 0])

    w.add_ssm_conv_kernel(D_CONV)
    w.add_ssm_inner_size(D_INNER)
    w.add_ssm_state_size(D_STATE)
    w.add_ssm_time_step_rank(N_V_HEADS)
    w.add_ssm_group_count(N_K_HEADS)
    w.add_key_value("qwen35.full_attention_interval", N_LAYER, GGUFValueType.UINT32)

    tokens = ["<unk>", "<s>", "</s>"] + [f"<0x{i:02X}>" for i in range(256)]
    n_vocab = len(tokens)
    w.add_tokenizer_model("llama")
    w.add_token_list(tokens)
    w.add_token_scores([0.0] * n_vocab)
    w.add_token_types([2, 3, 3] + [6] * 256)
    w.add_unk_token_id(0)
    w.add_bos_token_id(1)
    w.add_eos_token_id(2)

    # note: numpy shapes are the reverse of the ggml ne[] order
    w.add_tensor("token_embd.weight", rand(n_vocab, N_EMBD))
    w.add_tensor("output_norm.weight", np.ones(N_EMBD, dtype=np.float32))
    # output.weight omitted -> tied to token_embd

    for il in range(N_LAYER + 1):
        recurrent = il < N_LAYER and (il + 1) % N_LAYER != 0

        w.add_tensor(f"blk.{il}.attn_norm.weight",           np.ones(N_EMBD, dtype=np.float32))
        w.add_tensor(f"blk.{il}.post_attention_norm.weight", np.ones(N_EMBD, dtype=np.float32))

        if recurrent:
            w.add_tensor(f"blk.{il}.attn_qkv.weight",  rand(CONV_DIM, N_EMBD))
            w.add_tensor(f"blk.{il}.attn_gate.weight", rand(VALUE_DIM, N_EMBD))
            w.add_tensor(f"blk.{il}.ssm_conv1d.weight", rand(CONV_DIM, D_CONV))
            w.add_tensor(f"blk.{il}.ssm_dt.bias",  rand(N_V_HEADS))
            w.add_tensor(f"blk.{il}.ssm_a",        rng.uniform(-0.6, -0.2, size=N_V_HEADS).astype(np.float32))
            w.add_tensor(f"blk.{il}.ssm_beta.weight",  rand(N_V_HEADS, N_EMBD))
            w.add_tensor(f"blk.{il}.ssm_alpha.weight", rand(N_V_HEADS, N_EMBD))
            w.add_tensor(f"blk.{il}.ssm_norm.weight",  np.ones(D_STATE, dtype=np.float32))
            w.add_tensor(f"blk.{il}.ssm_out.weight",   rand(N_EMBD, VALUE_DIM))
        else:
            w.add_tensor(f"blk.{il}.attn_q.weight", rand(HEAD_K_DIM * N_HEAD * 2, N_EMBD))
            w.add_tensor(f"blk.{il}.attn_k.weight", rand(HEAD_K_DIM * N_HEAD_KV, N_EMBD))
            w.add_tensor(f"blk.{il}.attn_v.weight", rand(HEAD_K_DIM * N_HEAD_KV, N_EMBD))
            w.add_tensor(f"blk.{il}.attn_output.weight", rand(N_EMBD, HEAD_K_DIM * N_HEAD))
            w.add_tensor(f"blk.{il}.attn_q_norm.weight", np.ones(HEAD_K_DIM, dtype=np.float32))
            w.add_tensor(f"blk.{il}.attn_k_norm.weight", np.ones(HEAD_K_DIM, dtype=np.float32))

        w.add_tensor(f"blk.{il}.ffn_gate.weight", rand(N_FF, N_EMBD))
        w.add_tensor(f"blk.{il}.ffn_down.weight", rand(N_EMBD, N_FF))
        w.add_tensor(f"blk.{il}.ffn_up.weight",   rand(N_FF, N_EMBD))

    w.add_tensor(f"blk.{N_LAYER}.nextn.eh_proj.weight", rand(N_EMBD, 2 * N_EMBD))
    w.add_tensor(f"blk.{N_LAYER}.nextn.enorm.weight", np.ones(N_EMBD, dtype=np.float32))
    w.add_tensor(f"blk.{N_LAYER}.nextn.hnorm.weight", np.ones(N_EMBD, dtype=np.float32))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path} (n_vocab={n_vocab})")


if __name__ == "__main__":
    main()
