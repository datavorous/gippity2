import json
import torch
from pathlib import Path


class Config:
    def __init__(self, config_path="config.json"):
        with open(config_path, "r") as f:
            data = json.load(f)

        self.batch_size = data["data"]["batch_size"]
        self.block_size = data["data"]["block_size"]

        self.max_iters = data["training"]["max_iters"]
        self.eval_interval = data["training"]["eval_interval"]
        self.learning_rate = data["training"]["learning_rate"]
        self.eval_iters = data["training"]["eval_iters"]

        self.embed_dim = data["model"]["embed_dim"]
        self.num_heads = data["model"]["num_heads"]
        self.num_layers = data["model"]["num_layers"]
        self.dropout = data["model"]["dropout"]

        self.num_generate_tokens = data["generation"]["num_tokens"]

        self.seed = data["setup"]["seed"]
        self.device = "cuda"

        torch.manual_seed(self.seed)
