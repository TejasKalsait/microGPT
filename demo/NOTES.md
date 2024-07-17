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

## 