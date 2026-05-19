import os
import sys
import importlib.util
import torch
import torch.nn as nn
import math

# ---------------------------------------------------------------------------
# Re-use MultiHeadAttention from Phase 2
# ---------------------------------------------------------------------------
_here = os.path.dirname(os.path.abspath(__file__))

def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_here, fname))
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_mha_mod            = _load("mha_02", "02_multihead.py")
MultiHeadAttention  = _mha_mod.MultiHeadAttention


# ===========================================================================
# 1. Sinusoidal Positional Encoding
# ===========================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """
    Positional encoding from "Attention Is All You Need" (Vaswani 2017), §3.5.

      PE(pos, 2i)   = sin( pos / 10000^(2i / d_model) )
      PE(pos, 2i+1) = cos( pos / 10000^(2i / d_model) )

    The encoding is fixed (not learned) and added to the token embeddings
    before the first encoder block, giving the model a sense of word order.
    It is stored as a buffer so it moves with the model device but is NOT
    a trainable parameter — gradients will not flow through it.
    """

    def __init__(self, d_model, max_seq_len=512):
        super().__init__()
        self.d_model = d_model

        # ------------------------------------------------------------------
        # Build the (max_seq_len, d_model) encoding matrix from scratch.
        #
        # The exponent for each even dimension 2i is:
        #   2i / d_model  (ranges 0 → ~2)
        # Plugged into 10000^(…) and inverted gives the "division term":
        #   div_term[i] = 1 / 10000^(2i/d_model)
        #                = exp( -2i * ln(10000) / d_model )
        # ------------------------------------------------------------------

        # pos: column vector of position indices  shape (max_seq_len, 1)
        pos = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)

        # i: row vector of *half-dimension* indices  shape (1, d_model//2)
        # Each i corresponds to ONE sin/cos pair occupying dims 2i and 2i+1.
        i = torch.arange(0, d_model, 2, dtype=torch.float32).unsqueeze(0)

        # div_term: shape (1, d_model//2)
        # Broadcasting with pos (max_seq_len, 1) produces (max_seq_len, d_model//2)
        div_term = torch.exp(-i * math.log(10000.0) / d_model)

        # Angle matrix: (max_seq_len, d_model//2)
        # angle[pos, i] = pos / 10000^(2i/d_model)
        angle = pos * div_term   # broadcasts to (max_seq_len, d_model//2)

        # Allocate the full encoding matrix and fill even/odd columns
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(angle)   # even dims  → sin
        pe[:, 1::2] = torch.cos(angle)   # odd dims   → cos

        # Add a batch dimension so it broadcasts over (batch, seq, d_model)
        pe = pe.unsqueeze(0)   # (1, max_seq_len, d_model)

        # register_buffer: saved in state_dict, moved with .to(device),
        # but NOT listed in .parameters() → no gradient.
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Add positional encoding to input embeddings.

        Args:
            x : (batch, seq, d_model)
        Returns:
            (batch, seq, d_model) — same shape, positional signal injected
        """
        seq = x.size(1)
        # self.pe[:, :seq] slices the first `seq` positions,
        # shape (1, seq, d_model) — broadcasts over the batch dimension.
        return x + self.pe[:, :seq]

    def plot_encoding(self, n_pos=20, n_dim=16):
        """
        Print an ASCII heatmap of the encoding matrix.

        Rows  = positions 0 … n_pos-1
        Cols  = dimensions 0 … n_dim-1

        Character density maps value → visual intensity:
          [-1.0, -0.6)  →  " "  (very negative)
          [-0.6, -0.2)  →  "."
          [-0.2,  0.2)  →  ":"  (near zero)
          [ 0.2,  0.6)  →  "+"
          [ 0.6,  1.0]  →  "#"  (strongly positive)
        """
        density = [" ", ".", ":", "+", "#"]

        def char(v):
            # Map [-1, 1] linearly to bin index 0-4
            v = max(-1.0, min(1.0, float(v)))   # clamp
            idx = int((v + 1.0) / 2.0 * 4.999)  # scale to [0, 4.999]
            return density[idx]

        # pe is (1, max_seq_len, d_model); drop batch dim
        matrix = self.pe[0, :n_pos, :n_dim]

        print("=" * 60)
        print(f"Positional Encoding heatmap  ({n_pos} positions × {n_dim} dims)")
        print('  " " very negative  "." negative  ":" near-zero')
        print('  "+" positive       "#" very positive')
        print("=" * 60)

        # Column header: dim indices grouped by 4
        header = "pos | " + "".join(f"{d:<2}" for d in range(n_dim))
        print(header)
        print("-" * len(header))

        for p in range(n_pos):
            row = "".join(char(matrix[p, d]) + " " for d in range(n_dim))
            print(f" {p:2d} | {row}")

        print("=" * 60)


# ===========================================================================
# 2. Feed-Forward Network
# ===========================================================================

class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network from "Attention Is All You Need" §3.3.

      FFN(x) = max(0, x W_1 + b_1) W_2 + b_2

    "Position-wise" means the same two linear layers are applied independently
    to every token position — it is NOT a sequence-level operation.
    d_ff is typically 4 × d_model (e.g. 512 → 2048 in the base model).
    """

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)   # expand to inner dimension
        self.linear2 = nn.Linear(d_ff, d_model)   # project back to d_model
        self.relu    = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x : (batch, seq, d_model)
        Returns:
            (batch, seq, d_model)

        PyTorch applies nn.Linear to the last dimension only, so every token
        position is transformed by the same weights — hence "position-wise."
        """
        print(f"  FeedForward input  shape : {tuple(x.shape)}")

        x = self.linear1(x)   # (batch, seq, d_model) → (batch, seq, d_ff)
        x = self.relu(x)      # non-linearity (element-wise, shape unchanged)
        x = self.linear2(x)   # (batch, seq, d_ff) → (batch, seq, d_model)

        print(f"  FeedForward output shape : {tuple(x.shape)}")
        return x


# ===========================================================================
# 3. Encoder Block (Pre-LN)
# ===========================================================================

class EncoderBlock(nn.Module):
    """
    One encoder layer from "Attention Is All You Need" §3.1, but using
    Pre-LN (LayerNorm applied BEFORE the sublayer, not after).

    Standard Post-LN (paper):   x = LayerNorm( x + Sublayer(x) )
    Pre-LN  (this impl):        x = x + Sublayer( LayerNorm(x) )

    Pre-LN is more training-stable: gradients flow through the residual
    path without passing through a LayerNorm, so they don't vanish in deep stacks.
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # Sublayer 1: Multi-Head Self-Attention
        self.mha      = MultiHeadAttention(d_model, num_heads)

        # Sublayer 2: Position-wise Feed-Forward Network
        self.ff       = FeedForward(d_model, d_ff)

        # One LayerNorm before each sublayer (Pre-LN convention)
        self.norm1    = nn.LayerNorm(d_model)   # normalises before MHA
        self.norm2    = nn.LayerNorm(d_model)   # normalises before FFN

        # Dropout applied to each sublayer's output before the residual add.
        # This prevents co-adaptation among the neurons during training.
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x    : (batch, seq, d_model)  — token representations coming in
            mask : optional causal / padding mask passed through to MHA
        Returns:
            (batch, seq, d_model)  — enriched token representations

        Equations implemented here:
          z₁ = LayerNorm(x)                          ← Pre-LN: normalise first
          a  = MHA(z₁, z₁, z₁)                      ← self-attention (Q=K=V=z₁)
          x  = x + Dropout(a)                        ← 1st residual connection

          z₂ = LayerNorm(x)                          ← Pre-LN before FFN
          f  = FFN(z₂)                               ← feed-forward on each token
          x  = x + Dropout(f)                        ← 2nd residual connection
        """

        # ------------------------------------------------------------------
        # Sublayer 1 — Multi-Head Self-Attention with residual
        # ------------------------------------------------------------------

        z1 = self.norm1(x)           # Pre-LN: normalise input before MHA
                                     # Formula: z₁ = LayerNorm(x)

        a, _ = self.mha(z1, mask)    # Self-attention — Q, K, V all come from z₁
                                     # Formula: a = MHA(z₁, z₁, z₁)

        a = self.dropout(a)          # Dropout on the sublayer output (regularise)

        x = x + a                   # Residual: skip the whole MHA sublayer
                                     # Formula: x ← x + Dropout(a)
                                     # Keeps the gradient highway clean and direct

        # ------------------------------------------------------------------
        # Sublayer 2 — Position-wise Feed-Forward with residual
        # ------------------------------------------------------------------

        z2 = self.norm2(x)           # Pre-LN: normalise (now updated) x before FFN
                                     # Formula: z₂ = LayerNorm(x)

        f  = self.ff(z2)             # Feed-forward on each token independently
                                     # Formula: f = FFN(z₂)

        f  = self.dropout(f)         # Dropout on the FFN output

        x  = x + f                  # Residual: skip the whole FFN sublayer
                                     # Formula: x ← x + Dropout(f)

        return x                     # (batch, seq, d_model) enriched representations


# ===========================================================================
# 4. Residual analysis
# ===========================================================================

def residual_analysis(d_model=64, num_heads=4, d_ff=256, seq_len=6):
    """
    Pass a random input through one EncoderBlock and print the L2 norm at
    five checkpoints.  The residuals should keep magnitudes roughly stable
    even though each sublayer can add or subtract signal.
    """
    torch.manual_seed(0)

    block = EncoderBlock(d_model, num_heads, d_ff, dropout=0.0)
    block.eval()

    x = torch.randn(1, seq_len, d_model)

    def l2(t):
        # Mean L2 norm across the batch × position × dim tensor
        return t.norm(dim=-1).mean().item()

    print("=" * 60)
    print("Residual Analysis — L2 norms through EncoderBlock")
    print("=" * 60)
    print(f"  Config: d_model={d_model}, num_heads={num_heads}, d_ff={d_ff}, seq_len={seq_len}")
    print()

    with torch.no_grad():
        # --- Checkpoint 0: raw input ---
        print(f"  [0] Input                  L2 norm = {l2(x):.4f}")

        # --- Replicate the forward pass step by step ---
        z1   = block.norm1(x)
        a, _ = block.mha(z1)
        print(f"  [1] MHA output (a)         L2 norm = {l2(a):.4f}")

        x_after_mha = x + a
        print(f"  [2] After 1st residual     L2 norm = {l2(x_after_mha):.4f}")

        z2 = block.norm2(x_after_mha)
        f  = block.ff(z2)
        print(f"  [3] FFN output (f)         L2 norm = {l2(f):.4f}")

        x_final = x_after_mha + f
        print(f"  [4] After 2nd residual     L2 norm = {l2(x_final):.4f}")

    print()
    print("  Observation: residuals prevent magnitude collapse or explosion.")
    print("  The output norm stays in the same ballpark as the input norm,")
    print("  even though MHA and FFN transform the signal substantially.")
    print("=" * 60)


# ===========================================================================
# 5. Gradient flow through a stacked encoder
# ===========================================================================

def gradient_flow_test(d_model=64, num_heads=4, d_ff=256, seq_len=6):
    """
    Stack 2 EncoderBlocks on top of a learnable embedding layer, run a fake
    forward pass (MSE loss against a random target), backpropagate, and
    confirm that gradients reach the embedding layer.

    If gradients are zero or None at the embedding, the blocks are broken
    (vanishing gradients, detached computation graph, etc.).
    """
    torch.manual_seed(42)

    vocab_size = 100

    # Simulated embedding: maps integer token ids → d_model vectors
    embedding  = nn.Embedding(vocab_size, d_model)

    # Two stacked encoder blocks
    block1 = EncoderBlock(d_model, num_heads, d_ff, dropout=0.0)
    block2 = EncoderBlock(d_model, num_heads, d_ff, dropout=0.0)

    # Fake input: a batch of 1 sequence of `seq_len` random token ids
    token_ids = torch.randint(0, vocab_size, (1, seq_len))

    # Forward pass
    x = embedding(token_ids)           # (1, seq_len, d_model) — needs grad
    x.retain_grad()                    # ask PyTorch to keep ∂L/∂x (non-leaf)

    x = block1(x)                      # through block 1
    x = block2(x)                      # through block 2

    # Fake scalar loss: MSE against a random target of the same shape
    target = torch.randn_like(x)
    loss   = ((x - target) ** 2).mean()

    # Backward pass — computes gradients for every parameter in the graph
    loss.backward()

    # --- Report gradient norms ---
    print()
    print("=" * 60)
    print("Gradient Flow Test — 2 stacked EncoderBlocks")
    print("=" * 60)
    print(f"  Loss value : {loss.item():.6f}")
    print()

    emb_grad_norm = embedding.weight.grad.norm().item()
    print(f"  Embedding weight grad norm : {emb_grad_norm:.6f}")

    if emb_grad_norm > 0:
        print("  PASS — gradients reached the embedding layer.")
    else:
        print("  FAIL — gradients are zero; graph may be broken.")

    print()
    print("  Per-block parameter gradient norms:")
    for name, blk in [("Block 1", block1), ("Block 2", block2)]:
        for pname, p in blk.named_parameters():
            if p.grad is not None:
                print(f"    {name} | {pname:<30} grad norm = {p.grad.norm().item():.6f}")
    print("=" * 60)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    torch.manual_seed(42)

    D_MODEL   = 64
    NUM_HEADS = 4
    D_FF      = 256
    SEQ_LEN   = 6
    BATCH     = 2

    # -----------------------------------------------------------------------
    # STEP 1 — Sinusoidal Positional Encoding
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 1 — Sinusoidal Positional Encoding")
    print("=" * 70)

    pe_module = SinusoidalPositionalEncoding(D_MODEL, max_seq_len=512)

    # Confirm it is a buffer (not a parameter)
    param_names  = [n for n, _ in pe_module.named_parameters()]
    buffer_names = [n for n, _ in pe_module.named_buffers()]
    print(f"  Parameters : {param_names}  ← should be empty")
    print(f"  Buffers    : {buffer_names}  ← 'pe' should be here")
    print()

    # Verify forward adds PE to embeddings (shape unchanged)
    x_dummy = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    x_pe    = pe_module(x_dummy)
    print(f"  Input  shape : {tuple(x_dummy.shape)}")
    print(f"  Output shape : {tuple(x_pe.shape)}  ← must match input")
    print()

    # ASCII heatmap of the first 20 positions × 16 dims
    pe_module.plot_encoding(n_pos=20, n_dim=16)

    # -----------------------------------------------------------------------
    # STEP 2 — FeedForward
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2 — FeedForward (position-wise)")
    print("=" * 70)

    ff = FeedForward(D_MODEL, D_FF)
    x_ff_in  = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    print(f"  Calling FeedForward on (batch={BATCH}, seq={SEQ_LEN}, d_model={D_MODEL}):")
    with torch.no_grad():
        x_ff_out = ff(x_ff_in)
    print(f"  d_ff (inner dimension) = {D_FF}")
    print(f"  Same shape in and out confirms each position is handled independently.")

    # -----------------------------------------------------------------------
    # STEP 3 — EncoderBlock forward pass
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 3 — EncoderBlock (Pre-LN) forward pass")
    print("=" * 70)

    block = EncoderBlock(D_MODEL, NUM_HEADS, D_FF, dropout=0.0)
    block.eval()

    x_enc = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    print(f"  Input  : {tuple(x_enc.shape)}")
    with torch.no_grad():
        x_out = block(x_enc)
    print(f"  Output : {tuple(x_out.shape)}")
    print()
    n_params = sum(p.numel() for p in block.parameters())
    print(f"  EncoderBlock total parameters: {n_params:,}")

    # -----------------------------------------------------------------------
    # STEP 4 — Residual analysis
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 4 — Residual Analysis")
    print("=" * 70)

    residual_analysis(D_MODEL, NUM_HEADS, D_FF, SEQ_LEN)

    # -----------------------------------------------------------------------
    # STEP 5 — Gradient flow through 2 stacked blocks
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 5 — Gradient Flow (2 stacked EncoderBlocks)")
    print("=" * 70)

    gradient_flow_test(D_MODEL, NUM_HEADS, D_FF, SEQ_LEN)
