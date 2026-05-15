"""
Activation compression via Hadamard projectors for the modded-nanogpt speedrun repo.

Wraps per-transformer-layer computation in a custom autograd Function that
saves a compressed (low-rank) representation of the input activation instead
of the full (B, T, C) tensor. During backward, the approximate activation is
reconstructed and the layer is re-run.
"""
import torch
import torch.nn.functional as F

_compression_config = None


def _get_projector(seq_len, attn_dim, compression_rank):
    """Build piecewise projector for the given dimensions."""
    from instant_projectors import piecewise_coefficients_for  # noqa: F811

    coeffs, segment_len = piecewise_coefficients_for(
        kind="dct",  # DCT works for any seq_len (Hadamard needs power-of-2)
        seq_len=seq_len,
        rank=compression_rank,
    )
    return coeffs, segment_len


class CompressedLayer(torch.autograd.Function):
    """
    Custom autograd Function that wraps a single transformer layer.

    Forward: runs the layer normally, then compresses the input activation
    via Hadamard piecewise projection. Saves only the compressed representation.

    Backward: reconstructs approximate input from compressed form, re-runs
    the layer, and computes gradients via straight-through estimator (STE).
    """

    @staticmethod
    def forward(ctx, x, layer_fn, coeffs, segment_len):
        x_next = layer_fn(x)

        with torch.no_grad():
            from instant_projectors import piecewise_project

            x_proj = piecewise_project(x.detach(), coeffs, segment_len=segment_len)

        ctx.save_for_backward(x_proj, coeffs)
        ctx.layer_fn = layer_fn
        ctx.segment_len = segment_len
        ctx.x_shape = x.shape
        ctx.x_dtype = x.dtype
        return x_next

    @staticmethod
    def backward(ctx, grad_next):
        x_proj, coeffs = ctx.saved_tensors
        segment_len = ctx.segment_len
        B, T, C = ctx.x_shape

        coeffs_f = coeffs.to(dtype=torch.float32)
        x_proj_f = x_proj.to(dtype=torch.float32)

        segment_vals = torch.einsum("kr,brc->bkc", coeffs_f, x_proj_f)
        x_approx = segment_vals.repeat_interleave(segment_len, dim=1)
        x_approx = x_approx.to(dtype=ctx.x_dtype)
        x_approx = x_approx.detach().requires_grad_(True)

        with torch.enable_grad():
            x_next = ctx.layer_fn(x_approx)
        torch.autograd.backward(x_next, grad_next)
        return x_approx.grad, None, None, None


def make_layer_fn(x, i, resid_l, post_l, x0_inject, attn, norm_fn, attn_args,
                   qkvo_w, c_fc, c_proj, skip_gate_out, skip_connection,
                   x_backout_val):
    """
    Create a captured layer function for layer index `i`.

    Returns a closure that, given input `x_in`, runs the full layer
    (attention + MLP + residual scaling) and returns `x_out`.
    """
    resid_attn = resid_l[0]
    resid_mlp = resid_l[1]
    post_attn = post_l[0]
    post_mlp = post_l[1]

    def layer_fn(x_in):
        x = x_in
        if i == 6:
            x = x + skip_gate_out * skip_connection
        else:
            attn_in = x_backout_val if x_backout_val is not None else x
            attn_out = attn(norm_fn(attn_in), attn_args, qkvo_w)
            x = resid_attn * x + post_attn * attn_out + x0_inject
        x = resid_mlp * x + post_mlp * ReLUSqrdMLP(norm_fn(x), c_fc, c_proj)
        return x

    return layer_fn
