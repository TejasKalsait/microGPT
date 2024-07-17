import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import os

# Hyperparameters
train_split = 0.9
val_split = 1.0 - train_split

batch_size = 32
block_size = 8

loss_eval_iters = 200
loss_eval_interval = 500

learning_rate = 1e-3
epochs = 5000

tokens_to_generate = 500

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Training on -->", device)

# For regeneration
torch.manual_seed(800)

# Dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = '../data/shakespear.txt'
file_path = os.path.join(script_dir, relative_path)
with open(file_path, 'r') as f:
    text = f.read()

chrs = sorted(list(set(text)))
vocab_size = len(chrs)

# Mappings
stoi = {s:i for i, s in enumerate(chrs)}
itos = {i:s for i, s in enumerate(chrs)}

# Takes string and returns list of integers
encode = lambda context : [stoi[s] for s in context]
# Takes list of integers and returns string
decode = lambda ints : ''.join([itos[i] for i in ints])

# Creating dataset
data = torch.tensor(encode(text), dtype = torch.long)
n =  int(train_split * len(data))
train = data[:n]
val = data[n:]

# Returns a batch based on split
def get_batch(split):
    data = train if split == 'train' else val
    ix = torch.randint(len(data) - block_size, (batch_size,))
    xs = torch.stack([data[i : i+block_size] for i in ix])
    ys = torch.stack([data[i+1 : i+block_size+1] for i in ix])
    xs, ys = xs.to(device), ys.to(device)
    return xs, ys

# Estimate average loss for low noise
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(loss_eval_iters)
        for k in range(loss_eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Building the simple BiGram model
class BiGram(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        # Embedding layer which has it's __call__ function
        self.embedding_table = nn.Embedding(vocab_size, vocab_size)

    # The nn.Module handles the __call__ func
    def forward(self, idx, targets=None):

        # idx and targets shape (B, T)
        logits = self.embedding_table(idx)

        # Just in case we only want Logits while generating
        if targets is None:
            loss = None
        else:
            # Logits that come out have shape [B, T, C]. For every batch, There are 8 characters and within these 8 characters, every
            # charater is passed through the channels (lookup table) 
            B, T, C = logits.shape
            # The cross_entropy loss takes in logits with shape [B, C]
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # Takes in idx -> Past context of shape [B, T] and predicts and appends the next token in context
        
        # 
        for _ in range(max_new_tokens):
            # This will call the forward func
            logits, loss = self(idx)
            # Since it's bigram model, we care about the last timestep only, so we extract that
            logits = logits[:, -1, :]       # noe dim is [B, C]
            probs = F.softmax(logits, dim = -1)     # dim = -1 means last dim
            ix = torch.multinomial(probs, num_samples = 1)
            # Concatenating the next timestep 
            idx = torch.cat((idx, ix), dim = 1)
        return idx
    
# Initializing the model
model = BiGram()
m = model.to(device)

# Initializing the optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr = learning_rate)

# Training Loop

for epoch in range(epochs):

    # fetch a batch
    xb, yb = get_batch('train')

    # Forward pass
    logits, loss = m(xb, yb)

    # Backward pass with zero grad
    optimizer.zero_grad(set_to_none = True)
    loss.backward()

    # Update
    optimizer.step()

    # Evaluating loss once in a while
    if epoch % loss_eval_interval == 0:
        losses = estimate_loss()
        print(f"Epoch   {epoch} / {epochs}  Train Loss -->    {losses['train']:.4f}     Validation Loss -->    {losses['val']:.4f}")

# Generate Example from the model
context = torch.zeros((1, 1), dtype = torch.long, device = device)
print(decode(m.generate(context, max_new_tokens = tokens_to_generate)[0].tolist()))