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

n_embid = 32
head_size = 32

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

# Attention Head Block
class Head(nn.Module):
    '''
    A single self attention head
    input -> data of shape [B,T,C]
    output -> data of shape [B,T,head_size]
    '''

    def __init__(self, head_size) -> None:
        super().__init__()
        self.key = nn.Linear(n_embid, head_size, bias = False)
        self.query = nn.Linear(n_embid, head_size, bias = False)
        self.value = nn.Linear(n_embid, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        # Get key and query vectors
        k = self.key(x)     # [B,T,head_size]
        q = self.query(x)   # [B,T,head_size]
        # Compute attention scores
        wei = q @ k.transpose(-2, -1) * (C**-0.5)               # [B,T,head_size] @ [B,head_size, T] -> [B,T,T]
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))    # Same
        wei = F.softmax(wei, dim=-1)
        # Weighted aggregation of the values
        v = self.value(x)  # [B,T,head_size]
        out = wei @ v       # [B,T,T] @ [B,T,head_size] -> [B,T,head_size]
        return out
    
class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim = -1)
    
class FeedForward(nn.Module):

    def __init__(self, head_size) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(head_size, head_size),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


# Building the simple BiGram lamodel
class BiGram(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        # Embedding layer which has it's __call__ function
        self.embedding_table = nn.Embedding(vocab_size, n_embid)
        self.position_embedding_table = nn.Embedding(block_size, n_embid)
        self.sa_heads = MultiHeadAttention(4, head_size//4)
        self.ffwd = FeedForward(head_size)
        self.lm_head = nn.Linear(head_size, vocab_size)       # Right now the decoder to 65 vocab size

    # The nn.Module handles the __call__ func
    def forward(self, idx, targets=None):

        B, T = idx.shape
        # idx and targets shape (B, T)
        tok_emb = self.embedding_table(idx)     # [B, T, C]
        pos_embid = self.position_embedding_table(torch.arange(T, device = device))     # [T,C]
        x = tok_emb + pos_embid         # [B,T,C] + [T,C]
        x = self.sa_heads(x)            # [B,T,head_size]
        x = self.ffwd(x)                # [B, T, head_size]
        logits = self.lm_head(x)  # [B, T, vocab_size]

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

        for i in range(max_new_tokens):
            # crop idx to only have block_size of tokens incoming which is max of pos_encoding can take in feedforward
            idx_cropped = idx[:, -block_size:]
            # This will call the forward func
            logits, loss = self(idx_cropped)
            # Since it's bigram model, we care about the last timestep only, so we extract that
            logits = logits[:, -1, :]       # noe dim is [B, C]
            probs = F.softmax(logits, dim = -1)     # dim = -1 means last dim
            idx_next = torch.multinomial(probs, num_samples = 1)
            # Concatenating the next timestep 
            idx = torch.cat((idx, idx_next), dim = 1)
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