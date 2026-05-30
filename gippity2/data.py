import torch


class DataLoader:
    def __init__(self, file_path, device, batch_size, block_size):
        self.device = device
        self.batch_size = batch_size
        self.block_size = block_size

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

        encoded = [self.stoi[c] for c in text]
        data = torch.tensor(encoded, dtype=torch.long)

        split_idx = int(0.9 * len(data))
        self.train_data = data[:split_idx]
        self.val_data = data[split_idx:]

    def get_batch(self, split):
        data = self.train_data if split == "train" else self.val_data
        indices = torch.randint(len(data) - self.block_size, (self.batch_size,))

        x = torch.stack([data[i : i + self.block_size] for i in indices])
        y = torch.stack([data[i + 1 : i + self.block_size + 1] for i in indices])

        return x.to(self.device), y.to(self.device)

    def decode(self, indices):
        return "".join([self.itos[i] for i in indices])
