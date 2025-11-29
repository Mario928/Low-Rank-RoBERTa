# Low-Rank RoBERTa for AG News Classification: Achieving High Accuracy Under 1M Parameters

This repository contains our implementation for a project, where we develop a parameter-efficient RoBERTa model using Low-Rank Adaptation (LoRA) to achieve high accuracy on the AG News text classification dataset while staying under 1 million trainable parameters.

## Project Overview

We've implemented a RoBERTa variant with LoRA that balances model size and performance on text classification:

- Uses a parameter-efficient fine-tuning approach with only 0.XX million trainable parameters
- Implements Low-Rank Adaptation to modify only key, query, and value matrices
- Incorporates advanced text augmentation techniques for robust performance
- Achieves 94.4% accuracy on the test set

## Repository Structure

- `RoBERTa_LoRA_AGNews_Sub1M.ipynb`: Google Colab notebook containing all code for the project
- `README.md`: Project documentation

## Model Architecture

Our model uses RoBERTa-base with Low-Rank Adaptation:

- Base model: RoBERTa (12 transformer layers)
- LoRA rank: r = 2
- LoRA alpha: 4
- Target modules: query, key, and value matrices
- Dropout: 0.1
- Total trainable parameters: 704,260 (under 1 million)

## Training Methodology

We used the following training strategy:

- AdamW optimizer
- Learning rate: 2e-4
- Weight decay: 0.01
- Batch size: 64
- Training epochs: 3
- Hugging Face's Trainer API

## Data Augmentation

For improving model robustness, we applied multiple text augmentation techniques:

- Synonym replacement
- HTML entity insertion
- Character swapping
- Word duplication
- Case swapping
- Punctuation and spacing modifications
- Text truncation

## Text Preprocessing

We applied several preprocessing steps:

- Lowercasing the text
- Company name masking (replacing company names with [COMPANY] token)
- Number masking (replacing digits with [NUM] token)
- Whitespace normalization

## Results

Our model achieves:

- Test accuracy: 94.4%
- Trainable parameter count: 704,260  (well under the 1M limit)

## How to Run

1. Open `RoBERTa_LoRA_AGNews_Sub1M.ipynb` in Google Colab
2. Select "Runtime > Run all" to execute all cells
3. The notebook will:
   - Load and preprocess the AG News dataset
   - Apply text augmentation
   - Build and train the LoRA-adapted RoBERTa model
   - Evaluate on the test set
   - Visualize results with confusion matrices and learning curves

## Citations

Our implementation was inspired by:

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (Hu et al., 2021)
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) (Liu et al., 2019)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PEFT: Parameter-Efficient Fine-Tuning](https://github.com/huggingface/peft)

## Contributors

- Prashant Shihora
- Megh Panandikar
- Moulik Shah
