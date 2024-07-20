# microGPT

microGPT is a micro-scaled GPT model implemented from scratch with 10 million parameters. `Coded. From. Scratch.` This project demonstrates a comprehensive understanding of the transformer architecture, including self-attention and cross-attention mechanisms, layer normalization, and the overall structure of a GPT-like model.

## Project Overview

The goal of this project was to recreate and train a miniaturized version of GPT (Generative Pre-trained Transformer) from the ground up. This involved coding the entire model architecture, training process, and evaluation metrics from scratch. The project serves as a testament to the under-the-hood understanding of how transformer models function.

## Model Architecture

![model architecture image](https://github.com/TejasKalsait/microGPT/blob/main/images/mini_gpt_arch.jpg)

### Key Components
- **Self-Attention Mechanism**: Used to capture dependencies between different parts of the input sequence.
- **Feedforward Neural Networks**: Applied after the attention mechanism to process the representations.
- **Layer Normalization**: Implemented to stabilize and accelerate training.
- **Residual Connections**: Added to help the gradients flow through the network.

### Layers

- `input -> token_embedding + position_embedding -> input_embedding -> * [[self attention -> dropout -> layer_norm] *skip_connection -> @ [linear_mlp -> dropout -> layer_norm]] @skip_connection x 6 -> layer_norm -> linear_lm_head -> logits`

### Hyperparameters
- **Train Split**: 0.9
- **Validation Split**: 0.1
- **Batch Size**: 64
- **Block Size**: 256
- **Embedding Size**: 384
- **Number of Attention Heads**: 6
- **Number of Layers**: 6
- **Loss Evaluation Interval**: 500 epochs
- **Loss Evaluation Iterations**: 200
- **Learning Rate**: 3e-4
- **Epochs**: 5000
- **Dropout**: 0.2

### Results
- Parameters: 10,788,929
- `Train Loss: 1.1111`
- `Validation Loss: 1.4815`

## Notes

Detailed notes taken during the development and training of this model can be found in the [my notes](./demo/NOTES.md) file.

## Loss Record

The detailed record of training and validation losses at each phase can be found in the [loss records](./demo/Loss_record.md) file.

## Conclusion

`microGPT` is a robust miniaturized version of GPT, built and trained from scratch. It serves as a comprehensive demonstration of the inner workings of transformer models and their training processes. This project showcases the depth of understanding and practical implementation skills in the field of neural networks and deep learning.

## Reach out

Email - kalsaittejas10@gmail.com
[LinkedIn](https://www.linkedin.com/in/tkalsait/)  |  [Dreamer.ai](https://dreamer-ai.streamlit.app/)  |  [Portfolio](https://tejaskalsait.github.io/)
