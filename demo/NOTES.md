# Micro GPT

## Dataset

- The shakespear dataset contains `1,115,394` characters that makes up a vocab size of `65`

## Tokenizer and De-tokenizer

- There are many tokenizers and detokenizers out there that encode string into list of ints and list od ints back to string. There are many schemas to do this.
- For example, Google uses SentencePiece
![SentencePiece](https://github.com/google/sentencepiece), OpenAI uses Tiktoken
![Tiktoken](https://github.com/openai/tiktoken)
- We will just use normal mapping of string to int and vice versa for now.

- `The vocab_size and encoding output are inversly proportional`. If the vocab size is small, you wll have a longer sequence of encodings. For example "hii there" --> [46, 47, 47, 1, 58, 46, 43, 56, 43] in our character level tokanizer with vocab_size of 65. If you use the GPT-2 tokenizer Tiktoken, the encoind will be "hii there" --> [71, 4178, 612] since the vocab_size is 50,257.

## Creating the Batches

- dim 1 -> Batch dim
- dim 2 -> Block_size dim (8)
- So one training example is Input of [1, 8] and output of [1, 8].
- x[0] is the first character, y[0] is the expected output if feeded x[0].
- y[5] is the output expected when I feed x[0:5] and this output is also x[6] basically.
- Example - 
When input is [50] output is 42
When input is [50, 42] output is 1
When input is [50, 42, 1] output is 40
When input is [50, 42, 1, 40] output is 43
When input is [50, 42, 1, 40, 43] output is 1
When input is [50, 42, 1, 40, 43, 1] output is 53
When input is [50, 42, 1, 40, 43, 1, 53] output is 59
When input is [50, 42, 1, 40, 43, 1, 53, 59] output is 56

- `[B, T, C]` refers to Batch, Time, Channels dimensions.

## AdamW Optimizer

- An optimizer initialized with model parameters and learning rate and after doing losss.backward() you perform `optimizer.step()`

## Self-Attention Block

- Basically we want the tokens to talk to each other instead of Bigram taking only previous character info.

- Dimension -> [B, T, C] Batch, Time series, Channels

- If we take 5th token (Time series) we want this to talk to only the previous tokens and nothing in the future. (Becasue that needs to be predicted).

- The Simplest way to communicate is to take channels of this and all previous tokens and average them out to have an overall information for this particular token. Now this losses lots of spacial information and not good communicating. But a good start.

## Matrix multiplication as weighted aggregate of tokens

- We could use two for loops first going through batches and second going through Time dimensions.
- We then extract the required sub-torch ([b, :t+1]) and mean across the 1st dim and keep doing this.
- Obviously this method is super slow and we want to convert this to matrix multiplication.

- Suppose we take a A = [3, 3] matrix and multiply it with a B = [3, 2] matrix the result is a C = [3, 2] matrix.
- Now, the first value of the result came from first row of A and first column of B.
- Example -> 

A = [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]

B = [2, 7],
    [6, 4],
    [6, 5]

C = A @ B becomes   [14, 16],
                    [14, 16],
                    [14, 16]

`If any row of A is just ones, then C basically gets the sum across rows`

- Now we we change the A matrix be be `a triangle matrix`-

A = [1, 0, 0],
    [1, 1, 0],
    [1, 1, 1] using torch.tril(torch.ones(3,3))

C = A @ B becomes   [2, 7],
                    [8, 11],
                    [14, 16]

`8 and 11 are basically instead of summing across all the rows, only summing across that row and above`

- Now we we change the A matrix be be `a triangle matrix with mean across columns`-

A = [1.0, 0, 0],
    [0.5, 0.5, 0],
    [0.33, 0.33, 0.33] using torch.tril(torch.ones(3,3))

A = A / A.sum(1, keepdim = True)

C = A @ B becomes   [2.0, 7.0],
                    [4.0, 5.5],
                    [4.66, 5.33]

`4.0 and 5.5 are now the average of values in that row and above rows of B  -->> Exactly what we wanted`

<b>`So here the A matrix can be thought of like a MAsk or weight deciding how much value to give to which previous token`</b>

B, T, C = 4, 8, 3

Random trial batch of inputs of shape [B,T,C]
x = torch.randint(1, 65, (B, T, C)).float()     # B,T,C --> 4,8,3
print(x.shape)

A Lower triangle 1.0
tril = torch.tril(torch.ones(T, T))
Initializing weights to zero
wei = torch.zeros_like(tril).float()
Preventing Future tokens talking to current and past tokens
wei = wei.masked_fill(tril == 0, float('-inf'))
Normalizing to get averages across time dimension
wei = wei.softmax(dim = -1)
Talking of weights with the input
y = wei @ x     # [T,T] @ [B,T,C] -> [B,T,C]

`The weights are initialized to zero, then we prevent future tokens talking and then we do a softmax to get average. You see how you can tune the weights based on the input data to find out which past token is how much important to this current token and change the weights accordingly`

## Self Attention Block

