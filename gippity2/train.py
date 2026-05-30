import torch


class Trainer:
    def __init__(self, model, data_loader, learning_rate, device):
        self.model = model
        self.data_loader = data_loader
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    @torch.no_grad()
    def estimate_loss(self, eval_iters):
        self.model.eval()
        train_losses = []
        val_losses = []

        for _ in range(eval_iters):
            train_x, train_y = self.data_loader.get_batch("train")
            _, train_loss = self.model(train_x, train_y)
            train_losses.append(train_loss.item())

            val_x, val_y = self.data_loader.get_batch("val")
            _, val_loss = self.model(val_x, val_y)
            val_losses.append(val_loss.item())

        self.model.train()
        return {
            "train": sum(train_losses) / len(train_losses),
            "val": sum(val_losses) / len(val_losses),
        }

    def train(self, max_iters, eval_interval, eval_iters):
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model has {num_params} parameters")
        print(f"Training on {self.device}")

        for iteration in range(max_iters):
            if iteration % eval_interval == 0:
                losses = self.estimate_loss(eval_iters)
                print(
                    f"iter {iteration}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
                )

            x, y = self.data_loader.get_batch("train")
            logits, loss = self.model(x, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print("Training complete!")

    def generate(self, num_tokens):
        self.model.eval()
        context = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        output = self.model.generate(context, num_tokens)
        return self.data_loader.decode(output[0].tolist())

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}")
