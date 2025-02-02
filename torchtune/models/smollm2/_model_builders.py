from torchtune.models.llama3_2._component_builders import llama3_2

from torchtune.modules import TransformerDecoder


def smollm2_135m() -> TransformerDecoder:

    return llama3_2(
        vocab_size=49152,
        num_layers=30,
        num_heads=9,
        num_kv_heads=3,
        embed_dim=576,
        max_seq_len=8192,
        intermediate_dim=1536,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=100_000,
        scale_factor=32,
    )


def smollm2_360m() -> TransformerDecoder:

    return llama3_2(
        vocab_size=49152,
        num_layers=32,
        num_heads=15,
        num_kv_heads=5,
        embed_dim=960,
        max_seq_len=8192,
        intermediate_dim=2560,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=100_000,
        scale_factor=32,
    )


def smollm2_1_7b() -> TransformerDecoder:

    return llama3_2(
        vocab_size=49152,
        num_layers=24,
        num_heads=32,
        num_kv_heads=32,
        embed_dim=2048,
        max_seq_len=8192,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=100_000,
        scale_factor=32,
    )
