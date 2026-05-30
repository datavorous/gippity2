import torch
from gippity2 import GPT, DataLoader, Trainer, Config

config = Config('config.json')
config.device = 'cpu'

data_loader = DataLoader('input.txt', config.device, config.batch_size, config.block_size)
model = GPT(data_loader.vocab_size, config.embed_dim, config.num_heads, config.num_layers, config.block_size, config.dropout, config.device).to(config.device)

trainer = Trainer(model, data_loader, config.learning_rate, config.device)
trainer.load('model_weights.pth')

while True:
    prompt = input("Prompt: ")
    if prompt.lower() == 'quit':
        break

    tokens = torch.tensor([[data_loader.stoi.get(c, 0) for c in prompt]], dtype=torch.long, device=config.device)
    output = model.generate(tokens, config.num_generate_tokens)
    print(data_loader.decode(output[0].tolist()) + "\n")
