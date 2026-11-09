"""
custom_nli_model.py

Small Transformer NLI Model — built completely from scratch using PyTorch.
No pre-trained weights are loaded. All weights are randomly initialised
and trained by us from the ground up.

Architecture (BERT-style encoder):
  Token Embedding (learned)
  + Segment Embedding (learned)
  + Positional Embedding (learned)
  → N × TransformerEncoderLayer (our own implementation)
  → [CLS] pooling
  → Custom NLI classification head
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Focal Loss  
# ─────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss with Label Smoothing:
      FL(p_t) = -(1 - p_t)^gamma * log(p_t)
    label_smoothing prevents overconfidence by softening the hard targets
    (e.g. 0.1 -> targets become [0.05, 0.95] instead of [0, 1]).
    gamma=0 → standard CrossEntropy; gamma=2 → strong focus on hard examples.
    """
    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.1):
        super().__init__()
        self.gamma           = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, labels: torch.Tensor,
                weight: torch.Tensor = None) -> torch.Tensor:
        ce = F.cross_entropy(
            logits, labels,
            weight=weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce)                        # probability of correct class
        focal = (1.0 - pt) ** self.gamma * ce      # down-weight easy examples
        return focal.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Head Self Attention 
# ─────────────────────────────────────────────────────────────────────────────
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim  = embed_dim
        self.num_heads  = num_heads
        self.head_dim   = embed_dim // num_heads
        self.scale      = math.sqrt(self.head_dim)

        # single projection matrix for Q, K, V
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        B, T, D = x.shape

        # project to Q, K, V and split heads
        qkv = self.qkv_proj(x)                           # (B, T, 3D)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                # (3, B, H, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, H, T, T)

        if attention_mask is not None:
            # attention_mask: (B, T) → expand to (B, 1, 1, T)
            mask = (1 - attention_mask).unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask.bool(), float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)           # (B, H, T, head_dim)
        context = context.permute(0, 2, 1, 3).reshape(B, T, D)

        return self.out_proj(context)


# ─────────────────────────────────────────────────────────────────────────────
# Position-wise Feed-Forward Network 
# ─────────────────────────────────────────────────────────────────────────────
class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Single Transformer Encoder Block 
# ─────────────────────────────────────────────────────────────────────────────
class TransformerBlock(nn.Module):
    """
    Pre-LayerNorm transformer block (more stable than post-LN for small models).
    x → LayerNorm → Attention → residual
      → LayerNorm → FFN       → residual
    """
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn   = FeedForward(embed_dim, ffn_dim, dropout)

    def forward(self, x, attention_mask=None):
        # self-attention with residual
        x = x + self.attn(self.norm1(x), attention_mask)
        # feed-forward with residual
        x = x + self.ffn(self.norm2(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Embedding Layer  
# ─────────────────────────────────────────────────────────────────────────────
class NLIEmbedding(nn.Module):
    """
    Three learned embedding tables combined:
      token_embed(token_ids) + position_embed(positions) + segment_embed(segment_ids)
    """
    def __init__(self, vocab_size: int, embed_dim: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self.token_embed   = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.position_embed= nn.Embedding(max_len, embed_dim)
        self.segment_embed = nn.Embedding(2, embed_dim)   # 0=passage, 1=claim
        self.norm          = nn.LayerNorm(embed_dim)
        self.dropout       = nn.Dropout(dropout)

    def forward(self, input_ids, token_type_ids=None):
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)

        x = self.token_embed(input_ids) + self.position_embed(positions)

        if token_type_ids is not None:
            x = x + self.segment_embed(token_type_ids)

        return self.dropout(self.norm(x))


# ─────────────────────────────────────────────────────────────────────────────
# NLI Classification Head  
# ─────────────────────────────────────────────────────────────────────────────
class NLIHead(nn.Module):
    """
    3-layer MLP classification head.
    Accepts a pooled vector of size `in_dim` → outputs class logits.
    `in_dim` = embed_dim * 2 when CLS + mean pooling are concatenated.
    """
    def __init__(self, in_dim: int, num_labels: int, dropout: float = 0.25):
        super().__init__()
        hidden = in_dim * 2
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, in_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim // 2, num_labels),
        )

    def forward(self, pooled):
        return self.head(pooled)


# ─────────────────────────────────────────────────────────────────────────────
# Full Model
# ─────────────────────────────────────────────────────────────────────────────
class NarrativeNLIModel(nn.Module):
    """
    Our Small NLI Language Model — built fully from scratch.

    Hyperparameters (defaults = 'small' config):
        vocab_size      = 30522   
        embed_dim       = 256     
        num_heads       = 8      
        num_layers      = 4       
        ffn_dim         = 512     
        max_len         = 256     
        num_labels      = 2       
        dropout         = 0.3     
        label_smoothing = 0.1     

    Parameter count ~ 10.7M 
    """

    def __init__(
        self,
        vocab_size: int       = 30522,
        embed_dim: int        = 256,
        num_heads: int        = 8,
        num_layers: int       = 4,       
        ffn_dim: int          = 512,     
        max_len: int          = 256,
        num_labels: int       = 2,
        dropout: float        = 0.35,
        focal_gamma: float    = 2.0,
        label_smoothing: float = 0.10,
    ):
        super().__init__()

        self.focal_loss = FocalLoss(gamma=focal_gamma, label_smoothing=label_smoothing)

        self.config = dict(
            vocab_size=vocab_size, embed_dim=embed_dim, num_heads=num_heads,
            num_layers=num_layers, ffn_dim=ffn_dim, max_len=max_len,
            num_labels=num_labels, dropout=dropout, focal_gamma=focal_gamma,
            label_smoothing=label_smoothing,
        )

        # embedding layer (our own)
        self.embedding = NLIEmbedding(vocab_size, embed_dim, max_len, dropout)

        # stack of transformer blocks (our own)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        # final layer norm
        self.final_norm = nn.LayerNorm(embed_dim)

        # classification head — takes CLS ++ mean-pool vector (size = embed_dim * 2)
        self.nli_head = NLIHead(embed_dim * 2, num_labels, dropout=0.25)

        # initialise all weights from scratch
        self.apply(self._init_weights)

        total, trainable = self.count_parameters()
        print(f"NarrativeNLIModel — total params: {total:,}  trainable: {trainable:,}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def encode(self, input_ids, attention_mask, token_type_ids=None):
        """Shared encoder: token ids -> contextual vectors (B, T, D)."""
        x = self.embedding(input_ids, token_type_ids)
        for layer in self.layers:
            x = layer(x, attention_mask)
        return self.final_norm(x)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None, class_weights=None):
        x = self.encode(input_ids, attention_mask, token_type_ids)   # (B, T, D)

        # ── Dual pooling: CLS token  +  masked mean pooling ──────────────────
        cls_vec  = x[:, 0, :]                               # (B, D)

        mask_f   = attention_mask.unsqueeze(-1).float()     # (B, T, 1)
        mean_vec = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-9)  # (B, D)

        # concatenate → richer representation than CLS alone
        pooled = torch.cat([cls_vec, mean_vec], dim=-1)     # (B, 2D)

        # classify
        logits = self.nli_head(pooled)                      # (B, num_labels)

        loss = None
        if labels is not None:
            w = class_weights.to(logits.device) if class_weights is not None else None
            loss = self.focal_loss(logits, labels, weight=w)

        return {"loss": loss, "logits": logits}

    def count_parameters(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def save(self, path: str):
        import json
        from pathlib import Path
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), f"{path}/nli_weights.pt")
        with open(f"{path}/nli_config.json", "w") as f:
            json.dump(self.config, f, indent=2)
        print(f"saved model → {path}")

    def load_encoder(self, path: str):
        """Copy MLM-pretrained encoder weights (embeddings + blocks) into this
        model. The classification head stays randomly initialised."""
        state = torch.load(f"{path}/encoder.pt", map_location="cpu")
        missing, unexpected = self.load_state_dict(state, strict=False)
        assert not unexpected, unexpected
        assert all(k.startswith("nli_head") for k in missing), missing
        print(f"loaded pretrained encoder from {path} ({len(state)} tensors)")

    @classmethod
    def load(cls, path: str):
        import json
        with open(f"{path}/nli_config.json") as f:
            cfg = json.load(f)
        model = cls(**cfg)
        model.load_state_dict(torch.load(f"{path}/nli_weights.pt", map_location="cpu"))
        return model


# ─────────────────────────────────────────────────────────────────────────────
# Masked-language-model head (pretraining only)
# ─────────────────────────────────────────────────────────────────────────────
class MLMHead(nn.Module):
    """Dense -> GELU -> LayerNorm -> vocab logits, decoder tied to token embeddings."""
    def __init__(self, embed: nn.Embedding):
        super().__init__()
        d = embed.weight.shape[1]
        self.dense = nn.Linear(d, d)
        self.norm  = nn.LayerNorm(d)
        self.embed = embed
        self.bias  = nn.Parameter(torch.zeros(embed.weight.shape[0]))

    def forward(self, h):
        h = self.norm(F.gelu(self.dense(h)))
        return h @ self.embed.weight.t() + self.bias
