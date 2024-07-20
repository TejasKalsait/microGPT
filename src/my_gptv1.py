import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import os

### Hyperparameters
train_split = 0.9               # Ration of training data we need
val_split = 1.0 - train_split   # Remaining data is test data

batch_size = 64                 # How many examples to process at once
block_size = 256                  # How many tokens to process at once (Token size)
n_embid = 384                    # Number of embeddings for each token (channels)
n_heads = 6                     # How many self-attention heads in a multi-head self-attention layer (communication channels    )
n_layer = 6                     # How big of a consecutive stack of (multi-head-->MLP) we want

loss_eval_interval = 500        # Calculate loss after how many epochs
loss_eval_iters = 200           # How many samples to consider to get an average of loss going on

learning_rate = 3e-4            # Learning rate for the Optimizer
epochs = 5000                   # Number of epochs
dropout = 0.2                   # Ratio of neurons to dropout for regularization

tokens_to_generate = 10000      # How big of an answer to generate
relative_input_path = '../data/shakespear.txt'
relative_output_path = '../output/output.txt'

# ______________________________________________________

script_dir = os.path.dirname(os.path.abspath(__file__))
input_file_path = os.path.join(script_dir, relative_input_path)
output_file_path = os.path.join(script_dir, relative_output_path)

# CPU or GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Training on -->", device)

# For regeneration
torch.manual_seed(800)

# Dataset
with open(input_file_path, 'r') as f:
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
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        # Get key and query vectors
        k = self.key(x)     # [B,T,head_size]
        q = self.query(x)   # [B,T,head_size]
        # Compute attention scores
        wei = q @ k.transpose(-2, -1) * (C**-0.5)               # [B,T,head_size] @ [B,head_size, T] -> [B,T,T]
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))    # Same
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)         # randomly prevet some token to communicate
        # Weighted aggregation of the values
        v = self.value(x)  # [B,T,head_size]
        out = wei @ v       # [B,T,T] @ [B,T,head_size] -> [B,T,head_size]
        return out
    
class MultiHeadAttention(nn.Module):

    '''
    Multiple self attention heads and concatenating results
    input -> data of shape [B,T,C]
    output -> data of shape [B,T,C]
    '''

    def __init__(self, n_heads, head_size) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embid, n_embid)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.dropout(self.proj(out))
        return out
    
class FeedForward(nn.Module):

    '''
    A simple MLP to perform computation after self-attention
    input -> data of shape [B,T,C]
    output -> data of shape [B,T,C]
    '''

    def __init__(self, n_embid) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embid, 4 * n_embid),        # In the paper they suggest neurons in computation to be 4x of input
            nn.ReLU(),
            nn.Linear(4 * n_embid, n_embid),        # Linear transformation for skip connections to keep the same channels
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    

class Block(nn.Module):

    '''
    A Block that performs multihead attention (communication) followed by MLP feedforward (computation)
    # Takes in num_heads and n_embid as inputs to calculate head size
    input -> data of shape [B,T,n_embid or C]
    output -> data of shape [B,T,n_embid or C]
    '''

    def __init__(self, n_heads, n_embid) -> None:
        super().__init__()
        head_size = n_embid// n_heads
        self.sa_heads = MultiHeadAttention(n_heads, head_size)
        self.ffw = FeedForward(n_embid)
        self.ln1 = nn.LayerNorm(n_embid)        # Layer normalization to normalize all the channels (depth)
        self.ln2 = nn.LayerNorm(n_embid)        # of each token of each batch

    def forward(self, x):
        # x -> [B,T,C]
        x = x + self.sa_heads(self.ln1(x))        # Both have residual connections (computation and add previous to it). Backpropogation is handeled.
        x = x + self.ffw(self.ln2(x))             # Adding layernorm before the actual layer based on recent advancements of attention blocks.
        return x    # [B,T,C]



# Building the simple BiGram lamodel
class GPTLanguageModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        # Embedding layer which has it's __call__ function
        self.embedding_table = nn.Embedding(vocab_size, n_embid)
        self.position_embedding_table = nn.Embedding(block_size, n_embid)
        self.blocks = nn.Sequential(*[Block(n_embid = n_embid, n_heads = n_heads) for _ in range(n_layer)])     # The `*` unpacks the list into seperate arguments how Sequencial expects.
        self. ln_f = nn.LayerNorm(n_embid)                  # Last layer norm
        self.lm_head = nn.Linear(n_embid, vocab_size)       # Right now the decoder to 65 vocab size

    # The nn.Module handles the __call__ func
    def forward(self, idx, targets=None):

        B, T = idx.shape
        # idx and targets shape (B, T)
        tok_emb = self.embedding_table(idx)     # [B, T, C]
        pos_embid = self.position_embedding_table(torch.arange(T, device = device))     # [T,C]
        x = tok_emb + pos_embid         # [B,T,C] + [T,C]
        x = self.blocks(x)              # [B, T,C]
        x = self.ln_f(x)                # [B, T, C]
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
model = GPTLanguageModel()
m = model.to(device)

# Checking for parameters
n_parameters = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters in the model   -->   {n_parameters}")

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

# Printing the final loss after training
final_loss = estimate_loss()
print("Model Trained Successfully!!")
print(f"Train Loss -->    {losses['train']:.4f}     Validation Loss -->    {losses['val']:.4f}")

# Generate Example from the model
context = torch.zeros((1, 1), dtype = torch.long, device = device)
output = decode(m.generate(context, max_new_tokens = tokens_to_generate)[0].tolist())

# Print or file
if tokens_to_generate < 1001:
    print("Printing the output from the Language model....")
    print(output)
else:
    with open(output_file_path, 'w') as file:
        print(f"Saving the output from the model at {output_file_path}")
        file.write(output)

# _________________________________________________________YAYYYYYY!!!!____________________________________________________________________________________

'''
Tejas Kalsait
email: kalsaittejas10@gmail.com
LinkedIn: https://www.linkedin.com/in/tkalsait/
'''