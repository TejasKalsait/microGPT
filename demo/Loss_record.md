# Record of all the losses

## After adding 3 sequencial attention-feedforward blocks

Model Trained Successfully!!
Train Loss -->    2.2838     Validation Loss -->    `2.2877`

## After adding residual blocks and 4x neurons in inner computation MLP

Model Trained Successfully!!
Train Loss -->    2.0105     Validation Loss -->    `2.0843`

## Added the Layer Normalization after attention and mlp layers

Total number of parameters in the mode    -->   42305
Model Trained Successfully!!
Train Loss -->    2.0148     Validation Loss -->    `2.0750`

## Added a layer norm before LM head

Total number of parameters in the mode    -->   42369
Model Trained Successfully!!
Train Loss -->    1.9976     Validation Loss -->    `2.0675`

## Final Training with bigger network and dropouts

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

Total number of parameters in the model    -->   10,788,929
Model Trained Successfully!!
Train Loss -->    1.1111     Validation Loss -->    `1.4815`