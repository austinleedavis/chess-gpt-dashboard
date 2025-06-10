import einops
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformers import GPT2Config, GPT2LMHeadModel


def hooked_to_hf_config(hooked_cfg: HookedTransformerConfig):
    return GPT2Config(
        n_embd=hooked_cfg.d_model,
        n_head=hooked_cfg.n_heads,
        n_ctx=hooked_cfg.n_ctx,
        n_layer=hooked_cfg.n_layers,
        layer_norm_epsilon=hooked_cfg.eps,
        vocab_size=hooked_cfg.d_vocab,
        activation_function=hooked_cfg.act_fn,
        scale_attn_weights=True,
        scale_attn_by_inverse_layer_idx=hooked_cfg.scale_attn_by_inverse_layer_idx,
        # "architectures": ["GPT2LMHeadModel"],
    )


def convert_gpt2_weights(gpt2, cfg: HookedTransformerConfig):
    """Converts hf_model to HookedTransformer"""
    state_dict = {}

    state_dict["embed.W_E"] = gpt2.transformer.wte.weight
    state_dict["pos_embed.W_pos"] = gpt2.transformer.wpe.weight

    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = gpt2.transformer.h[l].ln_1.weight
        state_dict[f"blocks.{l}.ln1.b"] = gpt2.transformer.h[l].ln_1.bias

        # In GPT-2, q,k,v are produced by one big linear map, whose output is
        # concat([q, k, v])
        W = gpt2.transformer.h[l].attn.c_attn.weight
        W_Q, W_K, W_V = torch.tensor_split(W, 3, dim=1)
        W_Q = einops.rearrange(W_Q, "m (i h)->i m h", i=cfg.n_heads)
        W_K = einops.rearrange(W_K, "m (i h)->i m h", i=cfg.n_heads)
        W_V = einops.rearrange(W_V, "m (i h)->i m h", i=cfg.n_heads)

        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn.W_K"] = W_K
        state_dict[f"blocks.{l}.attn.W_V"] = W_V

        qkv_bias = gpt2.transformer.h[l].attn.c_attn.bias
        qkv_bias = einops.rearrange(
            qkv_bias,
            "(qkv index head)->qkv index head",
            qkv=3,
            index=cfg.n_heads,
            head=cfg.d_head,
        )
        state_dict[f"blocks.{l}.attn.b_Q"] = qkv_bias[0]
        state_dict[f"blocks.{l}.attn.b_K"] = qkv_bias[1]
        state_dict[f"blocks.{l}.attn.b_V"] = qkv_bias[2]

        W_O = gpt2.transformer.h[l].attn.c_proj.weight
        W_O = einops.rearrange(W_O, "(i h) m->i h m", i=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_O"] = W_O
        state_dict[f"blocks.{l}.attn.b_O"] = gpt2.transformer.h[l].attn.c_proj.bias

        state_dict[f"blocks.{l}.ln2.w"] = gpt2.transformer.h[l].ln_2.weight
        state_dict[f"blocks.{l}.ln2.b"] = gpt2.transformer.h[l].ln_2.bias

        W_in = gpt2.transformer.h[l].mlp.c_fc.weight
        state_dict[f"blocks.{l}.mlp.W_in"] = W_in
        state_dict[f"blocks.{l}.mlp.b_in"] = gpt2.transformer.h[l].mlp.c_fc.bias

        W_out = gpt2.transformer.h[l].mlp.c_proj.weight
        state_dict[f"blocks.{l}.mlp.W_out"] = W_out
        state_dict[f"blocks.{l}.mlp.b_out"] = gpt2.transformer.h[l].mlp.c_proj.bias
    state_dict["unembed.W_U"] = gpt2.lm_head.weight.T

    state_dict["ln_final.w"] = gpt2.transformer.ln_f.weight
    state_dict["ln_final.b"] = gpt2.transformer.ln_f.bias
    return state_dict
