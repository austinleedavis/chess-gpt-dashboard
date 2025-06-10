import os

import einops
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformers import GPT2Config, GPT2LMHeadModel


def hooked_to_gpt2(
    hooked_state_dict: dict, hooked_cfg: HookedTransformerConfig
) -> tuple[GPT2LMHeadModel, GPT2Config]:
    hf_config = GPT2Config(
        n_embd=hooked_cfg.d_model,
        n_head=hooked_cfg.n_heads,
        n_positions=hooked_cfg.n_ctx,
        n_layer=hooked_cfg.n_layers,
        layer_norm_epsilon=hooked_cfg.eps,
        vocab_size=hooked_cfg.d_vocab,
        activation_function=hooked_cfg.act_fn,
        scale_attn_weights=True,
        scale_attn_by_inverse_layer_idx=hooked_cfg.scale_attn_by_inverse_layer_idx,
        architectures=["GPT2LMHeadModel"],
    )

    hf_state_dict = {}
    hf_state_dict["transformer.wte.weight"] = hooked_state_dict["embed.W_E"]
    hf_state_dict["transformer.wpe.weight"] = hooked_state_dict["pos_embed.W_pos"]

    for l in range(hooked_cfg.n_layers):
        # fmt: off
        hf_state_dict[f"transformer.h.{l}.ln_1.weight"] = hooked_state_dict[f"blocks.{l}.ln1.w"]
        hf_state_dict[f"transformer.h.{l}.ln_1.bias"] = hooked_state_dict[f"blocks.{l}.ln1.b"]
        
        W_Q = hooked_state_dict[f"blocks.{l}.attn.W_Q"]
        W_K = hooked_state_dict[f"blocks.{l}.attn.W_K"]
        W_V = hooked_state_dict[f"blocks.{l}.attn.W_V"]
        W_Q = einops.rearrange(W_Q, "i m h -> m (i h)")
        W_K = einops.rearrange(W_K, "i m h -> m (i h)")
        W_V = einops.rearrange(W_V, "i m h -> m (i h)")
        hf_state_dict[f"transformer.h.{l}.attn.c_attn.weight"] = torch.cat([W_Q, W_K, W_V], dim=1)

        b_Q = hooked_state_dict.get(f"blocks.{l}.attn.b_Q", torch.zeros(hooked_cfg.d_model))
        b_K = hooked_state_dict.get(f"blocks.{l}.attn.b_K", torch.zeros(hooked_cfg.d_model))
        b_V = hooked_state_dict.get(f"blocks.{l}.attn.b_V", torch.zeros(hooked_cfg.d_model))
        hf_state_dict[f"transformer.h.{l}.attn.c_attn.bias"] = torch.cat([b_Q, b_K, b_V], dim=0)
        
        W_O = hooked_state_dict[f"blocks.{l}.attn.W_O"]
        hf_state_dict[f"transformer.h.{l}.attn.c_proj.weight"] = einops.rearrange(W_O, "i h m -> (i h) m")
        hf_state_dict[f"transformer.h.{l}.attn.c_proj.bias"] = hooked_state_dict.get(f"blocks.{l}.attn.b_O", torch.zeros(hooked_cfg.d_model))
        
        hf_state_dict[f"transformer.h.{l}.ln_2.weight"] = hooked_state_dict[f"blocks.{l}.ln2.w"]
        hf_state_dict[f"transformer.h.{l}.ln_2.bias"] = hooked_state_dict[f"blocks.{l}.ln2.b"]
        
        hf_state_dict[f"transformer.h.{l}.mlp.c_fc.weight"] = hooked_state_dict[f"blocks.{l}.mlp.W_in"]
        hf_state_dict[f"transformer.h.{l}.mlp.c_fc.bias"] = hooked_state_dict.get(f"blocks.{l}.mlp.b_in", torch.zeros(4*hooked_cfg.d_model))
        hf_state_dict[f"transformer.h.{l}.mlp.c_proj.weight"] = hooked_state_dict[f"blocks.{l}.mlp.W_out"]
        hf_state_dict[f"transformer.h.{l}.mlp.c_proj.bias"] = hooked_state_dict.get(f"blocks.{l}.mlp.b_out", torch.zeros(hooked_cfg.d_model))
        # fmt: on

    hf_state_dict["transformer.ln_f.weight"] = hooked_state_dict["ln_final.w"]
    hf_state_dict["transformer.ln_f.bias"] = hooked_state_dict["ln_final.b"]
    hf_state_dict["lm_head.weight"] = hooked_state_dict["unembed.W_U"].T

    hf_model = GPT2LMHeadModel(hf_config)
    hf_model.load_state_dict(hf_state_dict)
    return hf_model, hf_config


hooked_config = HookedTransformerConfig(
    **torch.load(
        "models/tf_lens_lichess_16layers_ckpt_no_optimizer_cfg.pth", weights_only=False
    )
)
hooked_state_dict = torch.load(
    "models/tf_lens_lichess_16layers_ckpt_no_optimizer.pth", weights_only=False
)
hf_model, hf_config = hooked_to_gpt2(hooked_state_dict, hooked_config)

hooked_model = HookedTransformer(hooked_config)
hooked_model.load_and_process_state_dict(hooked_state_dict, fold_ln=False)

sample_input = torch.tensor([[15, 6, 4, 27, 9, 0, 25, 10, 0, 7, 4, 19]]).to("cpu")
# sample_output = torch.tensor([[6, 4, 27, 9, 0, 27, 10, 0, 7, 4, 19, 28]])
hf_model_output = hf_model(sample_input).logits.argmax(dim=-1)
hooked_model_output = hooked_model(sample_input).argmax(dim=-1)

hf_model.save_pretrained("models/hf_lichess16layers")
print(hf_model_output)
print(hooked_model_output)

print()
