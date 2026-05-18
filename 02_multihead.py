import os
import sys
import importlib.util
import torch
import torch.nn as nn
import math

# ---------------------------------------------------------------------------
# Re-use scaled_dot_product_attention from Phase 1 rather than rewriting it.
# importlib lets us load a file by path even though the directory has spaces.
# ---------------------------------------------------------------------------
_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "attention_01",
    os.path.join(_here, "01_attention.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
scaled_dot_product_attention = _mod.scaled_dot_product_attention


# ---------------------------------------------------------------------------
# 1. Causal mask
# ---------------------------------------------------------------------------

def build_causal_mask(seq_len):
    """
    Build a lower-triangular boolean mask for autoregressive (causal) attention.

    Convention used HERE  →  True  = position is ALLOWED to attend
                             False = position is BLOCKED (it's a future token)

    Why causal masking?
      In a decoder, when generating token i, it must not be able to "see"
      tokens i+1, i+2, … (those haven't been produced yet).  Blocking the
      upper triangle enforces this constraint.

    Note on convention flip:
      scaled_dot_product_attention() uses the OPPOSITE convention internally
      (True = mask OUT).  So wherever we pass this mask into SDPA we invert
      it with ~mask, converting "allowed" → "block" for SDPA's masked_fill.

    Returns:
        mask : (seq_len, seq_len) bool tensor
               mask[i, j] = True  if query i CAN attend to key j  (j <= i)
               mask[i, j] = False if query i CANNOT attend to key j (j > i)
    """
    # torch.tril keeps the lower triangle (including diagonal) as True;
    # everything above the diagonal becomes False.
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool).tril()
    return mask


def print_causal_mask(mask):
    """Print the mask as a grid of 1s (allowed) and 0s (blocked)."""
    seq_len = mask.size(0)
    print("Causal mask  (1 = can attend, 0 = blocked future position)")
    print("             " + "  ".join(f"k{j}" for j in range(seq_len)))
    for i in range(seq_len):
        row = "  ".join("1" if mask[i, j] else "0" for j in range(seq_len))
        print(f"  query {i}  [ {row} ]")
    print()


# ---------------------------------------------------------------------------
# 2. Multi-Head Attention module
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention as described in "Attention Is All You Need" (Vaswani 2017).

    The key idea: instead of one big attention computation over d_model dimensions,
    split into num_heads smaller attention computations over d_k = d_model/num_heads
    dimensions each.  Each head can learn to attend to different aspects of the input
    (syntax, semantics, coreference, …).  Their outputs are concatenated and projected
    back to d_model.

    Parameter layout (no biases, matching the paper):
        W_q : (d_model, d_model)  — projects input to all heads' queries at once
        W_k : (d_model, d_model)  — projects input to all heads' keys at once
        W_v : (d_model, d_model)  — projects input to all heads' values at once
        W_o : (d_model, d_model)  — projects concatenated head outputs back to d_model

    Total parameters = 4 * d_model^2  (each matrix is d_model × d_model, no bias)
    """

    def __init__(self, d_model, num_heads):
        super().__init__()

        # Guard: d_model must divide evenly so every head gets the same width
        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads   # dimension per head

        # Single projection matrices that cover ALL heads at once.
        # We slice out individual heads later during the forward pass.
        # bias=False matches the original paper.
        self.W_q = nn.Linear(d_model, d_model, bias=False)   # query projection
        self.W_k = nn.Linear(d_model, d_model, bias=False)   # key projection
        self.W_v = nn.Linear(d_model, d_model, bias=False)   # value projection
        self.W_o = nn.Linear(d_model, d_model, bias=False)   # output projection

    def forward(self, x, mask=None):
        """
        Args:
            x    : (batch, seq, d_model)  — input token embeddings
            mask : (seq, seq) bool tensor from build_causal_mask()
                   True = allowed, False = blocked.
                   Will be inverted before passing to scaled_dot_product_attention.
        Returns:
            output      : (batch, seq, d_model)
            head_weights: list of num_heads tensors, each (batch, seq, seq)
        """
        batch, seq, _ = x.shape

        # ------------------------------------------------------------------
        # Project the full input to Q, K, V (all heads packed together).
        # Shape after projection: (batch, seq, d_model)
        # d_model = num_heads * d_k, so we'll split along the last dim next.
        # ------------------------------------------------------------------
        Q_full = self.W_q(x)   # (batch, seq, d_model)
        K_full = self.W_k(x)   # (batch, seq, d_model)
        V_full = self.W_v(x)   # (batch, seq, d_model)

        # ------------------------------------------------------------------
        # Reshape to expose the head dimension.
        #
        # .view(batch, seq, num_heads, d_k)
        #   splits the last axis (d_model) into (num_heads, d_k) sub-spaces,
        #   one slice per head — each head sees a d_k-wide projection.
        #
        # .transpose(1, 2)  →  (batch, num_heads, seq, d_k)
        #   moves the head axis to position 1 so we can index by head easily.
        # ------------------------------------------------------------------
        Q = Q_full.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        # Q: (batch, num_heads, seq, d_k)
        K = K_full.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        # K: (batch, num_heads, seq, d_k)
        V = V_full.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        # V: (batch, num_heads, seq, d_k)

        # ------------------------------------------------------------------
        # Prepare the mask for scaled_dot_product_attention.
        # SDPA convention: True = mask OUT (add -1e9).
        # Our causal mask convention: True = ALLOWED.
        # So we invert: ~mask turns allowed→block for SDPA.
        # We also unsqueeze to (1, 1, seq, seq) so it broadcasts over
        # (batch, num_heads, seq, seq) inside SDPA.
        # ------------------------------------------------------------------
        sdpa_mask = None
        if mask is not None:
            # ~mask: True where SDPA should block (upper triangle / future tokens).
            # One leading dim to broadcast over the batch; SDPA scores are (batch, seq, seq),
            # so (1, seq, seq) is enough — we don't need a head dim here because we loop
            # over heads one at a time and pass a plain (batch, seq, d_k) Q/K/V each time.
            sdpa_mask = (~mask).unsqueeze(0)   # (1, seq, seq)

        # ------------------------------------------------------------------
        # Run attention for each head independently (explicit loop).
        # Each head works on a d_k-dimensional subspace of the embeddings.
        # ------------------------------------------------------------------
        head_outputs  = []   # will collect (batch, seq, d_k) tensors
        head_weights  = []   # will collect (batch, seq, seq) tensors

        for h in range(self.num_heads):
            # Slice out head h across the batch
            # Q[:, h] → (batch, seq, d_k)
            Q_h = Q[:, h]
            K_h = K[:, h]
            V_h = V[:, h]

            # scaled_dot_product_attention expects (batch, seq, d_k)
            out_h, w_h = scaled_dot_product_attention(Q_h, K_h, V_h, mask=sdpa_mask)
            # out_h : (batch, seq, d_k)
            # w_h   : (batch, seq, seq)  — the attention weight matrix for head h

            head_outputs.append(out_h)
            head_weights.append(w_h)

        # ------------------------------------------------------------------
        # Concatenate all heads along the last dimension.
        # Stack along dim=-1 then merge: (batch, seq, num_heads * d_k) = (batch, seq, d_model)
        # ------------------------------------------------------------------
        # torch.stack → (batch, seq, num_heads, d_k)
        # .view(...)  → (batch, seq, d_model)  flattening the head + d_k dims
        concat = torch.stack(head_outputs, dim=2)   # (batch, seq, num_heads, d_k)
        concat = concat.view(batch, seq, self.d_model)  # (batch, seq, d_model)

        # ------------------------------------------------------------------
        # Final linear projection: mix information across heads.
        # This lets the model learn how to weight/combine what each head found.
        # ------------------------------------------------------------------
        output = self.W_o(concat)   # (batch, seq, d_model)

        return output, head_weights


# ---------------------------------------------------------------------------
# 3. compare_heads() — run MHA and analyse each head's attention pattern
# ---------------------------------------------------------------------------

def compare_heads(d_model=64, num_heads=4, seq_len=6):
    """
    Run MultiHeadAttention on a fixed input and compare what each head attends to.
    Prints:
      • Each head's attention weight matrix (2 heads per row, side by side)
      • Diagonal % (self-attention strength) vs off-diagonal % per head
    """
    torch.manual_seed(42)

    mha = MultiHeadAttention(d_model, num_heads)
    mha.eval()   # no dropout / batch-norm effects

    # One batch, seq_len tokens, each embedded as a d_model-dim vector
    x = torch.randn(1, seq_len, d_model)

    # Causal mask: token i can only attend to tokens 0..i
    causal_mask = build_causal_mask(seq_len)

    with torch.no_grad():
        output, head_weights = mha(x, mask=causal_mask)

    print("=" * 70)
    print(f"MultiHeadAttention — d_model={d_model}, num_heads={num_heads}, seq_len={seq_len}")
    print("=" * 70)
    print(f"Input  shape : {tuple(x.shape)}")
    print(f"Output shape : {tuple(output.shape)}")
    print()

    # ---- Print attention matrices two heads per row -------------------------
    # Each matrix is (batch=1, seq, seq); we drop the batch dim for display.
    matrices = [head_weights[h][0] for h in range(num_heads)]  # list of (seq, seq)

    col_w    = 6    # chars per cell value
    mat_w    = col_w * seq_len + 2   # total width of one matrix block

    print("Attention weight matrices per head (rows=query, cols=key)")
    print("Values shown to 2 d.p.  |  upper triangle is 0 due to causal mask\n")

    # Print in pairs: head 0+1, then head 2+3, etc.
    pairs = [(h, h + 1) for h in range(0, num_heads, 2)]
    for (ha, hb) in pairs:
        # Header line
        label_a = f"Head {ha}".center(mat_w)
        label_b = f"Head {hb}".center(mat_w) if hb < num_heads else ""
        print(f"  {label_a}    {label_b}")

        # Column-index row
        col_idx = "".join(f"  k{j}  " for j in range(seq_len))
        print(f"  {'':>3} {col_idx}    {'':>3} {col_idx}")

        for i in range(seq_len):
            def fmt_row(mat):
                # .item() converts the 0-dim tensor to a plain Python float for f-string formatting
                return "".join(f"{mat[i, j].item():6.2f}" for j in range(seq_len))

            row_a = fmt_row(matrices[ha])
            row_b = fmt_row(matrices[hb]) if hb < num_heads else ""
            print(f"  q{i}: {row_a}    q{i}: {row_b}")
        print()

    # ---- Diagonal (self-attention) vs off-diagonal analysis ----------------
    print("-" * 70)
    print("Self-attention analysis  (diagonal = token attending to itself)")
    print(f"{'Head':<8} {'Diag sum':>12} {'Off-diag sum':>14} {'Diag %':>10} {'Off-diag %':>12}")
    print("-" * 70)

    for h, mat in enumerate(matrices):
        # mat is (seq, seq) — each row is a probability distribution
        diag_sum     = mat.diagonal().sum().item()   # sum of self-attention weights
        total_sum    = mat.sum().item()              # equals seq (one distribution per row)
        offdiag_sum  = total_sum - diag_sum
        diag_pct     = 100 * diag_sum  / total_sum
        offdiag_pct  = 100 * offdiag_sum / total_sum
        print(f"  Head {h}   {diag_sum:>12.4f}   {offdiag_sum:>12.4f}   {diag_pct:>9.1f}%   {offdiag_pct:>10.1f}%")

    print()
    return mha, head_weights


# ---------------------------------------------------------------------------
# 4. Parameter count — break down by layer, verify total = 4 * d_model^2
# ---------------------------------------------------------------------------

def count_parameters(mha):
    """
    Print the number of trainable parameters in each projection matrix
    and verify the total equals 4 * d_model^2.
    """
    d_model = mha.d_model

    layers = [("W_q", mha.W_q), ("W_k", mha.W_k), ("W_v", mha.W_v), ("W_o", mha.W_o)]

    print("=" * 50)
    print("Parameter count breakdown")
    print("=" * 50)
    print(f"  d_model   = {d_model}")
    print(f"  num_heads = {mha.num_heads}")
    print(f"  d_k       = {mha.d_k}  (= d_model / num_heads)")
    print()
    print(f"  {'Layer':<8} {'Shape':>18} {'# params':>12}")
    print(f"  {'-'*8} {'-'*18} {'-'*12}")

    total = 0
    for name, layer in layers:
        # weight shape is (out_features, in_features) for nn.Linear
        shape  = tuple(layer.weight.shape)
        count  = layer.weight.numel()
        total += count
        print(f"  {name:<8} {str(shape):>18} {count:>12,}")

    expected = 4 * d_model ** 2
    print(f"  {'TOTAL':<8} {'':>18} {total:>12,}")
    print()
    print(f"  Expected (4 × d_model²) = 4 × {d_model}² = {expected:,}")
    match = "PASS" if total == expected else "FAIL"
    print(f"  Verification: {match}")
    print("=" * 50)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    SEQ_LEN   = 6
    D_MODEL   = 64
    NUM_HEADS = 4

    # --- Causal mask demo ---
    print("=" * 70)
    print("STEP 1 — Causal mask")
    print("=" * 70)
    causal_mask = build_causal_mask(SEQ_LEN)
    print_causal_mask(causal_mask)

    # --- Multi-head attention + head comparison ---
    print("STEP 2 — Multi-Head Attention forward pass + head comparison")
    mha, head_weights = compare_heads(D_MODEL, NUM_HEADS, SEQ_LEN)

    # --- Parameter breakdown ---
    print("STEP 3 — Parameter count")
    count_parameters(mha)
