import torch
import torch.nn as nn
from torch.nn import functional as F


class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(embed_dim, head_size)
        self.query = nn.Linear(embed_dim, head_size)
        self.value = nn.Linear(embed_dim, head_size)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        keys = self.key(x)
        queries = self.query(x)
        values = self.value(x)

        scores = queries @ keys.transpose(-2, -1) / (embed_dim**0.5)
        scores = scores.masked_fill(self.mask[:seq_len, :seq_len] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        return weights @ values


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size, dropout):
        super().__init__()
        head_size = embed_dim // num_heads
        self.heads = nn.ModuleList(
            [
                AttentionHead(embed_dim, head_size, block_size, dropout)
                for _ in range(num_heads)
            ]
        )
        self.output = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.output(out)
        return self.dropout(out)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, block_size, dropout)
        self.feedforward = FeedForward(embed_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feedforward(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(
        self, vocab_size, embed_dim, num_heads, num_layers, block_size, dropout, device
    ):
        super().__init__()
        self.device = device
        self.block_size = block_size

        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.position_embed = nn.Embedding(block_size, embed_dim)

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(embed_dim, num_heads, block_size, dropout)
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.output = nn.Linear(embed_dim, vocab_size)

    def forward(self, token_ids, targets=None):
        batch_size, seq_len = token_ids.shape

        token_emb = self.token_embed(token_ids)
        pos_ids = torch.arange(seq_len, device=self.device)
        pos_emb = self.position_embed(pos_ids)

        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.norm(x)
        logits = self.output(x)

        loss = None
        if targets is not None:
            logits_flat = logits.view(batch_size * seq_len, -1)
            targets_flat = targets.view(batch_size * seq_len)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def generate(self, token_ids, num_new_tokens):
        for _ in range(num_new_tokens):
            token_ids_trimmed = token_ids[:, -self.block_size :]
            logits, _ = self(token_ids_trimmed)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_ids = torch.cat([token_ids, next_token], dim=1)

        return token_ids
