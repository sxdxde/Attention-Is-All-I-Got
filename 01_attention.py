import torch
import torch.nn.functional as F
import math

# ---------------------------------------------------------------------------
# Core attention function
# ---------------------------------------------------------------------------

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Manual implementation of scaled dot-product attention from "Attention Is All You Need".

    The idea: each token (query) looks at every other token (key) and decides
    how much to "attend" to each one.  The result is a weighted blend of values.

    Args:
        Q    : (batch, seq_q, d_k)  — what each token is looking for
        K    : (batch, seq_k, d_k)  — what each token is advertising / offering
        V    : (batch, seq_k, d_v)  — the actual content to pass forward if attended to
        mask : (batch, seq_q, seq_k) or broadcastable
               True/1 at position [b, i, j] means query i is NOT allowed to
               attend to key j (used for causal / padding masks).
    Returns:
        output      : (batch, seq_q, d_v)  — new token representations
        attn_weights: (batch, seq_q, seq_k) — the learned attention distribution
    """

    # d_k is the query/key dimension; we need it for the scaling factor
    d_k = Q.size(-1)

    # ------------------------------------------------------------------
    # Step 1: Raw alignment scores
    #   Q @ K^T  →  (batch, seq_q, seq_k)
    #   Entry [b, i, j] = dot product of query_i and key_j.
    #   A large positive value means query i and key j are "compatible."
    # ------------------------------------------------------------------
    scores = Q @ K.transpose(-2, -1)

    # ------------------------------------------------------------------
    # Step 2: Scale by 1 / sqrt(d_k)
    #   Without this, dot products grow in magnitude as d_k increases,
    #   pushing softmax into saturation (near-zero gradients → slow training).
    #   Scaling keeps the variance of the scores roughly constant at 1.
    # ------------------------------------------------------------------
    scores = scores / math.sqrt(d_k)

    # ------------------------------------------------------------------
    # Step 3: Optional mask (additive, not multiplicative)
    #   We add a very large negative number to forbidden positions so that
    #   exp(-1e9) ≈ 0 after softmax — effectively zeroing those weights.
    #   Common uses:
    #     • Causal mask: prevent token i from seeing tokens after it
    #     • Padding mask: ignore padding tokens in variable-length batches
    # ------------------------------------------------------------------
    if mask is not None:
        scores = scores.masked_fill(mask.bool(), -1e9)

    # ------------------------------------------------------------------
    # Step 4: Softmax — convert raw scores into a probability distribution
    #   dim=-1 means we normalise across the KEY axis (columns), so each
    #   query token ends up with a distribution over all key tokens.
    #   After this step: all values ≥ 0 and each row sums to exactly 1.
    # ------------------------------------------------------------------
    attn_weights = F.softmax(scores, dim=-1)

    # ------------------------------------------------------------------
    # Step 5: Weighted sum of value vectors
    #   attn_weights @ V  →  (batch, seq_q, d_v)
    #   Each output token is a convex combination of the value vectors,
    #   weighted by how much attention was paid to each key position.
    # ------------------------------------------------------------------
    output = attn_weights @ V

    # Return both so callers can inspect the attention distribution
    return output, attn_weights


# ---------------------------------------------------------------------------
# Verification — print every intermediate shape and sanity-check row sums
# ---------------------------------------------------------------------------

def verify():
    print("=" * 60)
    print("VERIFY: scaled dot-product attention")
    print("=" * 60)

    # Fixed seed guarantees the same random tensors every run
    torch.manual_seed(42)

    # Small, human-readable dimensions: 1 example in the batch,
    # 5 tokens in the sequence, 4-dimensional key/query/value vectors
    batch, seq, d_k = 1, 5, 4

    # In a real transformer these would come from learned linear projections
    # of the token embeddings (W_Q, W_K, W_V weight matrices).
    # Here we use random normal vectors just to exercise the math.
    Q = torch.randn(batch, seq, d_k)   # queries
    K = torch.randn(batch, seq, d_k)   # keys
    V = torch.randn(batch, seq, d_k)   # values

    print(f"\nInputs")
    print(f"  Q shape : {tuple(Q.shape)}  — (batch, seq_q, d_k)")
    print(f"  K shape : {tuple(K.shape)}  — (batch, seq_k, d_k)")
    print(f"  V shape : {tuple(V.shape)}  — (batch, seq_k, d_v)")

    # --- Recompute each step manually so we can label the intermediate shapes ---

    # Q @ K^T : each of the seq_q queries dot-producted with each of the seq_k keys
    scores_raw = Q @ K.transpose(-2, -1)
    print(f"\nIntermediate tensors")
    print(f"  scores_raw  = Q @ K^T        : {tuple(scores_raw.shape)}  — (batch, seq_q, seq_k)")

    # Divide every score by sqrt(d_k) to prevent softmax saturation
    scores_scaled = scores_raw / math.sqrt(d_k)
    print(f"  scores_scaled / sqrt({d_k})      : {tuple(scores_scaled.shape)}  — same shape, values shrunk")

    # Softmax over the key axis — turns scores into a valid probability distribution
    attn_weights = F.softmax(scores_scaled, dim=-1)
    print(f"  attn_weights (softmax)       : {tuple(attn_weights.shape)}  — (batch, seq_q, seq_k)")

    # Weighted average of value vectors using the attention distribution
    output = attn_weights @ V
    print(f"  output = attn_weights @ V    : {tuple(output.shape)}  — (batch, seq_q, d_v)")

    # Print the raw attention matrix so we can see token-to-token relationships
    print("\nAttention weight matrix  (rows = query tokens, cols = key tokens):")
    w = attn_weights[0]   # drop batch dimension → (seq, seq)
    for i, row in enumerate(w):
        vals = "  ".join(f"{v:.4f}" for v in row)
        print(f"  row {i}: [{vals}]")

    # Verify the fundamental property: softmax rows must sum to 1
    # atol=1e-5 allows for minor floating-point rounding
    row_sums = attn_weights.sum(dim=-1)
    print(f"\nRow sums (should all be 1.0): {row_sums[0].tolist()}")
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), \
        "Attention rows do not sum to 1!"
    print("Assertion passed — every row sums to 1.0")

    # Return the (seq, seq) matrix for the visualiser
    return attn_weights[0]


# ---------------------------------------------------------------------------
# Visualiser — formatted grid with per-row max marked with an asterisk
# ---------------------------------------------------------------------------

def visualize_attention(attn_weights, tokens):
    """
    Print a human-readable attention grid.

    Args:
        attn_weights : (seq, seq) tensor — rows are queries, cols are keys
        tokens       : list of strings of length seq (the token labels)
    """
    seq = len(tokens)
    assert attn_weights.shape == (seq, seq), \
        f"Expected ({seq},{seq}), got {tuple(attn_weights.shape)}"

    # Make every column wide enough to fit the longest token name
    col_w = max(len(t) for t in tokens) + 2

    print("\n" + "=" * 60)
    print("ATTENTION VISUALISATION")
    print("rows = query (what's attending)  |  cols = key (what's attended to)")
    print("* marks the highest-weight key for each query token")
    print("=" * 60)

    # --- Header row: column labels (key/attended-to tokens) ---
    header = " " * (col_w + 2)   # indent to align with row labels
    for t in tokens:
        header += t.center(col_w)
    print(header)
    print(" " * (col_w + 2) + "-" * (col_w * seq))

    # --- One row per query token ---
    for i, row_token in enumerate(tokens):
        row = attn_weights[i]

        # Find which key position this query attends to most
        max_idx = row.argmax().item()

        # Build the line: row label | attention values (max marked with *)
        line = row_token.rjust(col_w) + " |"
        for j, val in enumerate(row):
            cell = f"{val:.2f}"
            # Append * for the peak attention position, space otherwise
            cell = cell + ("*" if j == max_idx else " ")
            line += cell.center(col_w)
        print(line)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # --- Part 1: verify correctness with generic random tensors ---
    verify()

    # --- Part 2: re-run with word tokens so the grid is interpretable ---
    tokens = ["the", "cat", "sat", "on", "mat"]
    seq, d_k = len(tokens), 4

    # seed=42 matches verify() so the weights are identical — same Q, K, V
    torch.manual_seed(42)
    Q = torch.randn(1, seq, d_k)
    K = torch.randn(1, seq, d_k)
    V = torch.randn(1, seq, d_k)

    # Run the full attention computation; discard the output, keep the weights
    _, attn_weights = scaled_dot_product_attention(Q, K, V)

    # attn_weights is (batch=1, seq, seq); drop the batch dim for the visualiser
    visualize_attention(attn_weights[0], tokens)

    print("""
============================================================
SUMMARY — what each tensor shape means
============================================================

Q / K / V  →  (batch=1, seq=5, d_k=4)
    Each of the 5 tokens is represented as a 4-dimensional
    vector.  In a real transformer these come from learned
    linear projections of the token embeddings.

scores_raw  →  (1, 5, 5)
    Entry [b, i, j] = dot product of query token i with key
    token j.  A high value means token i "finds token j
    relevant."  One row per query token, one column per key.

scores_scaled  →  (1, 5, 5)
    Same shape.  Dividing by sqrt(d_k) keeps the dot products
    from growing large as d_k increases — large values push
    softmax into regions with near-zero gradients, making
    training unstable.

attn_weights  →  (1, 5, 5)
    After softmax each row is a valid probability distribution
    (all values ≥ 0, row sums = 1.0).  Row i tells you how
    much token i attends to every other token in the sequence.

output  →  (1, 5, 4)
    For each query token, a weighted average of the value
    vectors.  Tokens that received high attention weights
    contribute more to the output representation.

WHY rows sum to 1
    softmax(x)_j = exp(x_j) / Σ_k exp(x_k)
    By construction the denominator normalises the numerators,
    so Σ_j softmax(x)_j = 1 for every row.  This makes the
    output a convex combination of the value vectors — a proper
    weighted average, not an unbounded sum.
============================================================
""")
