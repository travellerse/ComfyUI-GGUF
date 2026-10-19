# Fused Triton dequant for GGUF Q5_K blocks.
#
# The torch path in dequant.py expands every 256-value block through reshapes, broadcasts and
# separate fp16 elementwise kernels: 26-30 GB/s measured on gfx1030 (RX 6700 XT), against
# 206-441 GB/s for a plain copy of the same bytes. This kernel does it in one pass and reaches
# 280-506 GB/s, i.e. it is bandwidth-bound rather than instruction-bound.
#
# Bit-exactness is the requirement, not an optimisation: a different rounding here silently changes
# the model's weights. Two things must match the torch path exactly:
#   * PyTorch computes half arithmetic in fp32 (opmath) but materialises *each* tensor op in the
#     output dtype, so d*scale, d*sc-min and the subtraction round separately. One fused expression
#     looks plausible and is wrong.
#   * d/dmin are stored as fp16 in the block regardless of the output dtype and widened afterwards.
#
# Validated in the Triton tree at docs/rdna2/performance/q5k_triton_dequant.py: bit-exact for fp16
# and bf16 over block counts including non-multiples of the tile, and over random-bit blocks where
# d/dmin are NaN or Inf.

import os

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:  # the fallback below is the torch path, so this is not fatal
    _HAS_TRITON = False

Q5K_BLOCK_BYTES = 176
Q5K_QK_K = 256

# Off by default: enable with GGUF_TRITON_DEQUANT=1
TRITON_DEQUANT = os.environ.get("GGUF_TRITON_DEQUANT", "0") == "1"

if _HAS_TRITON:

    @triton.jit
    def _deq5k(B, Out, n_blocks, BPP: tl.constexpr, BB: tl.constexpr, KK: tl.constexpr,
               OT: tl.constexpr):
        pid = tl.program_id(0)
        blk = pid * BPP + tl.arange(0, BPP)[:, None]
        v = tl.arange(0, KK)[None, :]
        valid = blk < n_blocks
        base = blk * BB

        g = v >> 5
        c = v & 31

        d_bits = tl.load(B + base + 0, mask=valid, other=0).to(tl.uint16) | (
            tl.load(B + base + 1, mask=valid, other=0).to(tl.uint16) << 8)
        dm_bits = tl.load(B + base + 2, mask=valid, other=0).to(tl.uint16) | (
            tl.load(B + base + 3, mask=valid, other=0).to(tl.uint16) << 8)
        d = d_bits.to(tl.float16, bitcast=True).to(OT)
        dmin = dm_bits.to(tl.float16, bitcast=True).to(OT)

        qs = tl.load(B + base + 48 + (v >> 6) * 32 + c, mask=valid, other=0).to(tl.int32)
        qh = tl.load(B + base + 16 + c, mask=valid, other=0).to(tl.int32)
        lo = (qs >> ((g & 1) * 4)) & 15
        hi = (qh >> g) & 1
        q = (lo | (hi << 4)).to(tl.float32)

        s_d = tl.load(B + base + 4 + (g & 3), mask=valid, other=0).to(tl.int32)
        s_m = tl.load(B + base + 8 + (g & 3), mask=valid, other=0).to(tl.int32)
        s_md = tl.load(B + base + 12 + (g & 3), mask=valid, other=0).to(tl.int32)
        sc = tl.where(g < 4, s_d & 63, (s_md & 15) | ((s_d >> 2) & 48))
        mv = tl.where(g < 4, s_m & 63, (s_md >> 4) | ((s_m >> 2) & 48))

        dscale = (d.to(tl.float32) * sc.to(tl.float32)).to(OT)
        minscale = (dmin.to(tl.float32) * mv.to(tl.float32)).to(OT)
        prod = (dscale.to(tl.float32) * q).to(OT)
        out = (prod.to(tl.float32) - minscale.to(tl.float32)).to(OT)

        tl.store(Out + blk * KK + v, out, mask=valid)


_TL_OT = {torch.float16: "float16", torch.bfloat16: "bfloat16"}
stats = {"applied": 0, "fell_back": 0}


def triton_dequantize_q5k(blocks, dtype, block_d=4, num_warps=4):
    """(n_blocks, 176) uint8 -> (n_blocks, 256) in `dtype`, or None if not applicable."""
    if not (TRITON_DEQUANT and _HAS_TRITON):
        return None
    if dtype not in _TL_OT or blocks.dtype != torch.uint8 or not blocks.is_cuda:
        stats["fell_back"] += 1
        return None
    n_blocks = blocks.shape[0]
    try:
        out = torch.empty(n_blocks * Q5K_QK_K, device=blocks.device, dtype=dtype)
        _deq5k[(triton.cdiv(n_blocks, block_d),)](
            blocks, out, n_blocks, BPP=block_d, BB=Q5K_BLOCK_BYTES, KK=Q5K_QK_K,
            OT=getattr(tl, _TL_OT[dtype]), num_warps=num_warps)
    except Exception as exc:  # never let a fast path break a load
        stats["fell_back"] += 1
        print(f"[GGUF] triton Q5_K dequant failed, using torch path: {exc}")
        return None
    stats["applied"] += 1
    return out.reshape(n_blocks, Q5K_QK_K)

# --- Q6_K ---
#
# Same structure as the Q5_K path and the same exactness contract: d is an fp16 field inside the
# block (bytes 208-209) that is widened to the output dtype, and d*scale rounds to the output dtype
# before being multiplied by the (integer) quantized value, because PyTorch materialises every
# elementwise op in the output dtype.
#
# Validated in the Triton tree at docs/rdna2/performance/q6k_triton_dequant.py: bit-exact against
# this plugin's own dequantize_blocks_Q6_K for fp16 and bf16 over block counts 1..4097 including
# non-multiples of the tile, with random-bit blocks where d is NaN or Inf and again with d forced
# finite. Compare NaN/Inf positions rather than using torch.equal, which reports an identical
# decode containing NaN as unequal.

Q6K_BLOCK_BYTES = 210
Q6K_QK_K = 256

if _HAS_TRITON:

    @triton.jit
    def _deq6k(B, Out, n_blocks, BPP: tl.constexpr, BB: tl.constexpr, KK: tl.constexpr,
               OT: tl.constexpr):
        pid = tl.program_id(0)
        blk = pid * BPP + tl.arange(0, BPP)[:, None]
        i = tl.arange(0, KK)[None, :]
        valid = blk < n_blocks
        base = blk * BB

        half = i // 128           # which 64-byte ql group / 32-byte qh group
        qh_byte = base + 128 + half * 32 + (i % 32)
        shift = 2 * ((i % 128) // 32)
        hi = (tl.load(B + qh_byte, mask=valid, other=0).to(tl.int32) >> shift) & 3

        ql_byte = base + half * 64 + (i % 64)
        nib = (i % 128) // 64
        lo = (tl.load(B + ql_byte, mask=valid, other=0).to(tl.int32) >> (nib * 4)) & 15

        q = ((lo | (hi << 4)).to(tl.int32) - 32).to(tl.float32)

        d_bits = tl.load(B + base + 208, mask=valid, other=0).to(tl.uint16) | (
            tl.load(B + base + 209, mask=valid, other=0).to(tl.uint16) << 8)
        d = d_bits.to(tl.float16, bitcast=True).to(OT)
        sc = tl.load(B + base + 192 + (i // 16), mask=valid, other=0).to(tl.int8).to(OT)

        dscale = (d.to(tl.float32) * sc.to(tl.float32)).to(OT)
        out = (dscale.to(tl.float32) * q).to(OT)
        tl.store(Out + blk * KK + i, out, mask=valid)


def triton_dequantize_q6k(blocks, dtype, block_d=4, num_warps=4):
    """(n_blocks, 210) uint8 -> (n_blocks, 256) in `dtype`, or None if not applicable."""
    if not (TRITON_DEQUANT and _HAS_TRITON):
        return None
    if dtype not in _TL_OT or blocks.dtype != torch.uint8 or not blocks.is_cuda:
        stats["fell_back"] += 1
        return None
    n_blocks = blocks.shape[0]
    try:
        out = torch.empty(n_blocks * Q6K_QK_K, device=blocks.device, dtype=dtype)
        _deq6k[(triton.cdiv(n_blocks, block_d),)](
            blocks, out, n_blocks, BPP=block_d, BB=Q6K_BLOCK_BYTES, KK=Q6K_QK_K,
            OT=getattr(tl, _TL_OT[dtype]), num_warps=num_warps)
    except Exception as exc:  # never let a fast path break a load
        stats["fell_back"] += 1
        print(f"[GGUF] triton Q6_K dequant failed, using torch path: {exc}")
        return None
    stats["applied"] += 1
    return out.reshape(n_blocks, Q6K_QK_K)
