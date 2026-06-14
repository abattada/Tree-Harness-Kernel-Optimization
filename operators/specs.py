"""Operator registry: 16 operators with PyTorch references, input builders,
tolerances, and bytes-moved models for roofline analysis.

Contract for LLM-generated candidates (enforced by harness/eval_one.py):
the candidate module must define

    def triton_run(*inputs) -> torch.Tensor

with the exact input order documented in each spec's `signature_doc`.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F


@dataclass
class OperatorSpec:
    name: str
    category: str                       # elementwise | reduction | normalization | loss | matmul | misc
    make_inputs: Callable               # (seed:int, device:str) -> tuple[Tensor,...]
    ref: Callable                       # (*inputs) -> Tensor
    rtol: float
    atol: float
    bytes_moved: Callable               # (inputs, output) -> int   (ideal DRAM traffic)
    signature_doc: str                  # shown to the LLM
    forbidden_substrings: list = field(default_factory=list)  # naive anti-cheat
    compute_bound: bool = False         # roofline anchor only applies when False


def _t(numel_bytes: int) -> int:
    return int(numel_bytes)


def _nbytes(*tensors) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)


def _gen(seed: int, device: str):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return g


def _randn(shape, seed, device, dtype=torch.float32):
    g = _gen(seed, device)
    return torch.randn(*shape, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)


# ---------------------------------------------------------------- elementwise

def _vector_add_inputs(seed, device):
    n = 16_000_000
    return (_randn((n,), seed, device), _randn((n,), seed + 1, device))


def _vector_exp_inputs(seed, device):
    return (_randn((16_000_000,), seed, device),)


def _geglu_inputs(seed, device):
    return (_randn((8192, 8192), seed, device),)   # second half is the gate


def _geglu_ref(x):
    a, b = x.chunk(2, dim=-1)
    return F.gelu(a, approximate="tanh") * b


def _swiglu_ref(x):
    a, b = x.chunk(2, dim=-1)
    return F.silu(a) * b


# ----------------------------------------------------------------- reduction

def _sum_inputs(seed, device):
    return (_randn((33_554_432,), seed, device),)


def _welford_inputs(seed, device):
    return (_randn((8192, 4096), seed, device),)


def _welford_ref(x):
    mean = x.mean(dim=-1)
    var = x.var(dim=-1, unbiased=False)
    return torch.stack([mean, var])     # (2, n_rows)


# ------------------------------------------------------------- normalization

def _rows_inputs(seed, device, rows=8192, cols=4096):
    return (_randn((rows, cols), seed, device),)


def _layer_norm_ref(x):
    return F.layer_norm(x, normalized_shape=(x.shape[-1],), eps=1e-5)


def _rms_norm_ref(x):
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-5)


def _softmax_ref(x):
    return torch.softmax(x, dim=-1)


# -------------------------------------------------------------------- loss

def _ce_inputs(seed, device):
    logits = _randn((8192, 32768), seed, device)
    g = _gen(seed + 7, device)
    targets = torch.randint(0, 32768, (8192,), generator=g).to(device)
    return (logits, targets)


def _ce_ref(logits, targets):
    return F.cross_entropy(logits, targets)


def _flce_inputs(seed, device):
    x = _randn((4096, 2048), seed, device)
    w = _randn((32768, 2048), seed + 1, device) * 0.02
    g = _gen(seed + 7, device)
    targets = torch.randint(0, 32768, (4096,), generator=g).to(device)
    return (x, w, targets)


def _flce_ref(x, w, targets):
    return F.cross_entropy(x @ w.t(), targets)


def _kl_inputs(seed, device):
    logp = torch.log_softmax(_randn((8192, 8192), seed, device), dim=-1)
    q = torch.softmax(_randn((8192, 8192), seed + 1, device), dim=-1)
    return (logp, q)


def _kl_ref(logp, q):
    return F.kl_div(logp, q, reduction="batchmean")


# ------------------------------------------------------------------ matmul

def _gemm_inputs(seed, device):
    a = _randn((4096, 4096), seed, device, dtype=torch.float16)
    b = _randn((4096, 4096), seed + 1, device, dtype=torch.float16)
    return (a, b)


def _gemm_ref(a, b):
    return a @ b


def _addmm_inputs(seed, device):
    bias = _randn((4096,), seed + 2, device, dtype=torch.float16)
    a = _randn((4096, 4096), seed, device, dtype=torch.float16)
    b = _randn((4096, 4096), seed + 1, device, dtype=torch.float16)
    return (bias, a, b)


def _addmm_ref(bias, a, b):
    return torch.addmm(bias, a, b)


def _int4_gemm_inputs(seed, device):
    m, k, n = 4096, 4096, 4096
    a = _randn((m, k), seed, device, dtype=torch.float16)
    g = _gen(seed + 3, device)
    w_packed = torch.randint(-(2 ** 31), 2 ** 31 - 1, (k // 8, n), generator=g,
                             dtype=torch.int64).to(torch.int32).to(device)
    scales = (torch.rand(n, generator=_gen(seed + 4, device)) * 0.01 + 0.005).to(
        device=device, dtype=torch.float16)
    return (a, w_packed, scales)


def _int4_gemm_ref(a, w_packed, scales):
    k = a.shape[1]
    shifts = torch.arange(0, 32, 4, device=a.device, dtype=torch.int32)
    w = (w_packed.unsqueeze(1) >> shifts.view(1, -1, 1)) & 0xF        # (K//8, 8, N)
    w = w.reshape(k, -1).to(torch.float16)
    w = (w - 8.0) * scales                                            # (K, N)
    return a @ w


# -------------------------------------------------------------------- misc

def _rope_inputs(seed, device):
    b, h, s, d = 8, 32, 4096, 128
    x = _randn((b, h, s, d), seed, device, dtype=torch.float16)
    pos = torch.arange(s, dtype=torch.float32)
    inv = 1.0 / (10000 ** (torch.arange(0, d // 2, dtype=torch.float32) / (d // 2)))
    ang = torch.outer(pos, inv)
    return (x, ang.cos().to(device, torch.float16), ang.sin().to(device, torch.float16))


def _rope_ref(x, cos, sin):
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    c = cos.view(1, 1, *cos.shape)
    s = sin.view(1, 1, *sin.shape)
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


def _embedding_inputs(seed, device):
    weight = _randn((131072, 1024), seed, device)
    g = _gen(seed + 5, device)
    idx = torch.randint(0, 131072, (1_048_576,), generator=g).to(device)
    return (weight, idx)


def _embedding_ref(weight, idx):
    return weight[idx]


# ------------------------------------------------------------------ registry

def _spec(name, category, make_inputs, ref, rtol, atol, bytes_moved, sig,
          forbidden=None, compute_bound=False):
    return OperatorSpec(
        name=name, category=category, make_inputs=make_inputs, ref=ref,
        rtol=rtol, atol=atol, bytes_moved=bytes_moved, signature_doc=sig,
        forbidden_substrings=forbidden or [], compute_bound=compute_bound)


OPERATORS: dict[str, OperatorSpec] = {}

for s in [
    _spec("vector_add", "elementwise", _vector_add_inputs, lambda x, y: x + y,
          1e-5, 1e-5, lambda i, o: _nbytes(*i, o),
          "triton_run(x: f32[16M], y: f32[16M]) -> f32[16M]  # elementwise x+y"),
    _spec("vector_exp", "elementwise", _vector_exp_inputs, torch.exp,
          1e-5, 1e-5, lambda i, o: _nbytes(*i, o),
          "triton_run(x: f32[16M]) -> f32[16M]  # elementwise exp(x)",
          forbidden=[r"torch\.exp"]),
    _spec("geglu", "elementwise", _geglu_inputs, _geglu_ref,
          1e-4, 1e-4, lambda i, o: _nbytes(*i, o),
          "triton_run(x: f32[8192, 8192]) -> f32[8192, 4096]"
          "  # a,b = x.chunk(2,-1); gelu_tanh(a)*b",
          forbidden=[r"(F|functional)\.gelu"]),
    _spec("swiglu", "elementwise", _geglu_inputs, _swiglu_ref,
          1e-4, 1e-4, lambda i, o: _nbytes(*i, o),
          "triton_run(x: f32[8192, 8192]) -> f32[8192, 4096]"
          "  # a,b = x.chunk(2,-1); silu(a)*b",
          forbidden=[r"(F|functional)\.silu"]),
    _spec("sum", "reduction", _sum_inputs, torch.sum,
          1e-3, 1e-1, lambda i, o: _nbytes(*i),
          "triton_run(x: f32[32M]) -> f32[] (0-dim scalar tensor)  # sum(x); "
          "ALL reduction stages must be Triton kernels — torch.sum/.sum() "
          "anywhere (including combining partials) is rejected",
          forbidden=[r"torch\.sum", r"(?<!tl)\.sum\("]),
    _spec("welford", "reduction", _welford_inputs, _welford_ref,
          1e-3, 1e-3, lambda i, o: _nbytes(*i, o),
          "triton_run(x: f32[8192, 4096]) -> f32[2, 8192]"
          "  # row 0 = per-row mean, row 1 = per-row population variance",
          forbidden=[r"torch\.(mean|var)\b", r"(?<!tl)\.(mean|var)\("]),
    _spec("layer_norm", "normalization", _rows_inputs, _layer_norm_ref,
          1e-4, 1e-4, lambda i, o: _nbytes(*i, o),
          "triton_run(x: f32[8192, 4096]) -> f32[8192, 4096]"
          "  # layernorm over last dim, eps=1e-5, no affine",
          forbidden=[r"(torch|F|functional)\.layer_norm"]),
    _spec("rms_norm", "normalization", _rows_inputs, _rms_norm_ref,
          1e-4, 1e-4, lambda i, o: _nbytes(*i, o),
          "triton_run(x: f32[8192, 4096]) -> f32[8192, 4096]"
          "  # x * rsqrt(mean(x^2, -1) + 1e-5), no affine",
          forbidden=[r"(torch|F|functional)\.rms_norm"]),
    _spec("softmax", "normalization", _rows_inputs, _softmax_ref,
          1e-4, 1e-4, lambda i, o: _nbytes(*i, o),
          "triton_run(x: f32[8192, 4096]) -> f32[8192, 4096]  # softmax over last dim",
          forbidden=[r"(torch|F|functional)\.(log_)?softmax"]),
    _spec("cross_entropy", "loss", _ce_inputs, _ce_ref,
          1e-3, 1e-3, lambda i, o: _nbytes(i[0]),
          "triton_run(logits: f32[8192, 32768], targets: i64[8192]) -> f32[]"
          "  # mean cross-entropy",
          forbidden=[r"(torch|F|functional)\.(cross_entropy|log_softmax|nll_loss)"]),
    _spec("fused_linear_cross_entropy", "loss", _flce_inputs, _flce_ref,
          1e-2, 1e-2, lambda i, o: _nbytes(*i),
          "triton_run(x: f32[4096, 2048], w: f32[32768, 2048], targets: i64[4096])"
          " -> f32[]  # mean CE of (x @ w.T) without materializing full logits",
          forbidden=[r"(torch|F|functional)\.(cross_entropy|nll_loss)"], compute_bound=True),
    _spec("kl_div", "loss", _kl_inputs, _kl_ref,
          1e-3, 1e-3, lambda i, o: _nbytes(*i),
          "triton_run(log_p: f32[8192, 8192], q: f32[8192, 8192]) -> f32[]"
          "  # KLDiv: sum(q*(log q - log_p)) / batch (batchmean)",
          forbidden=[r"(torch|F|functional)\.kl_div"]),
    _spec("gemm", "matmul", _gemm_inputs, _gemm_ref,
          2e-2, 2e-2, lambda i, o: _nbytes(*i, o),
          "triton_run(a: f16[4096, 4096], b: f16[4096, 4096]) -> f16[4096, 4096]  # a @ b",
          forbidden=[r"torch\.(mm|matmul|addmm|bmm|einsum)\b", r"[\w\)\]]\s*@\s*[\w\(]"], compute_bound=True),
    _spec("addmm", "matmul", _addmm_inputs, _addmm_ref,
          2e-2, 2e-2, lambda i, o: _nbytes(*i, o),
          "triton_run(bias: f16[4096], a: f16[4096, 4096], b: f16[4096, 4096])"
          " -> f16[4096, 4096]  # bias + a @ b",
          forbidden=[r"torch\.(mm|matmul|addmm|bmm|einsum)\b", r"[\w\)\]]\s*@\s*[\w\(]"], compute_bound=True),
    _spec("int4_gemm", "matmul", _int4_gemm_inputs, _int4_gemm_ref,
          3e-2, 3e-2, lambda i, o: _nbytes(*i, o),
          "triton_run(a: f16[4096, 4096], w_packed: i32[512, 4096], scales: f16[4096])"
          " -> f16[4096, 4096]  # w[k,n] = ((w_packed[k//8,n] >> (4*(k%8))) & 0xF - 8)"
          " * scales[n]; return a @ w (dequantize inside the kernel)",
          forbidden=[r"torch\.(mm|matmul|addmm|bmm|einsum)\b", r"[\w\)\]]\s*@\s*[\w\(]"], compute_bound=True),
    _spec("rope", "misc", _rope_inputs, _rope_ref,
          2e-2, 2e-2, lambda i, o: _nbytes(i[0], o),
          "triton_run(x: f16[8, 32, 4096, 128], cos: f16[4096, 64], sin: f16[4096, 64])"
          " -> f16[8, 32, 4096, 128]  # rotate-half RoPE: out = cat(x1*cos - x2*sin,"
          " x1*sin + x2*cos) where x1,x2 = split(x, 64, dim=-1)"),
    _spec("embedding", "misc", _embedding_inputs, _embedding_ref,
          0.0, 0.0, lambda i, o: _nbytes(i[1], o),
          "triton_run(weight: f32[131072, 1024], idx: i64[1048576]) -> f32[1048576, 1024]"
          "  # gather rows: out[i] = weight[idx[i]]",
          forbidden=[r"(F|functional)\.embedding", r"index_select", r"\[\s*idx"]),
]:
    OPERATORS[s.name] = s


def get_spec(name: str) -> OperatorSpec:
    return OPERATORS[name]


def ref_source(spec: OperatorSpec) -> str:
    """Reference implementation source for the prompt (best effort)."""
    try:
        return inspect.getsource(spec.ref)
    except (OSError, TypeError):
        return f"# builtin: {spec.ref}"
