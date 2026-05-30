import torch
from gippity2 import GPT, DataLoader, Trainer, Config

config = Config("config.json")

data_loader = DataLoader(
    file_path="input.txt",
    device=config.device,
    batch_size=config.batch_size,
    block_size=config.block_size,
)

model = GPT(
    vocab_size=data_loader.vocab_size,
    embed_dim=config.embed_dim,
    num_heads=config.num_heads,
    num_layers=config.num_layers,
    block_size=config.block_size,
    dropout=config.dropout,
    device=config.device,
).to(config.device)

trainer = Trainer(model, data_loader, config.learning_rate, config.device)
trainer.train(config.max_iters, config.eval_interval, config.eval_iters)

print("Generating text:")
generated_text = trainer.generate(config.num_generate_tokens)
print(generated_text)
