import torch
from gippity2 import GPT, DataLoader, Trainer, Config

config = Config('config.json')
config.device = 'cpu'

data_loader = DataLoader(
    file_path='input.txt',
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
trainer.load('model_weights.pth')

print("Model loaded. Enter text prompts to generate continuations.")
print("Type 'quit' to exit.\n")

while True:
    prompt = input("Enter prompt: ")
    if prompt.lower() == 'quit':
        break

    prompt_tokens = [data_loader.stoi.get(c, 0) for c in prompt]
    context = torch.tensor([prompt_tokens], dtype=torch.long, device=config.device)

    model.eval()
    with torch.no_grad():
        output = model.generate(context, config.num_generate_tokens)

    generated = data_loader.decode(output[0].tolist())
    print(f"Generated: {generated}\n")
