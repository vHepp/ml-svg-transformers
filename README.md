# Transformer Scaling Laws on SVGs

**Author:** Vincent Hepola

**Course:** CS-GY 6923 (Spring 2026) - New York University Tandon School of Engineering

## Project Overview

This repository contains the code and experiments for investigating language model scaling laws on Scalable Vector Graphics (SVG) code. The project trains custom decoder-only Transformer models from scratch on a curated dataset of SVG paths. It primarily explores the differences between **Standard Parameterization (SP)** and  **Maximal Update Parameterization (µP)** , demonstrating how µP enables zero-shot hyperparameter transfer across model scales. Finally, the project evaluates the model's ability to learn strict XML syntax and mathematical coordinate structures through conditional and unconditional generation.

## Repository Structure

The experimental pipeline is divided into three main Jupyter Notebooks, which should generally be run in the following order:

### 1. `main.ipynb` (Data Preparation & Scaling Sweeps)

This notebook serves as the foundation for the project. It handles data preprocessing, model definitions, and the core scaling law experiments.

* **Data Pipeline:** Downloads the `starvector/svg-stack-simple` dataset via Hugging Face, applies regex-based cleaning, filters out sequences by length (50-2048 characters), and tokenizes the data using a custom BPE tokenizer (vocab size: 1,000).
* **Model Architectures:** Defines the `CustomTransformer` (Standard Parameterization) and `MuCustomTransformer` (Maximal Update Parameterization) architectures.
* **Learning Rate Sweeps:** Conducts extensive learning rate sweeps on the proxy "Tiny" model to identify optimal learning rates.
* **Scaling Laws:** Trains a family of models across various parameter counts ("Tiny" to "XL") to plot the power-law decay of validation loss and compares the hyperparameter stability of SP versus µP.

### 2. `train.ipynb` (Full Model Training)

This notebook is dedicated to the full, prolonged training runs of the scaled-up models, particularly utilizing the µP framework.

* **Training Loop:** Executes the training loop for the "XL" model with 44M parameters.
* **Metrics Tracking:** Captures detailed training and validation loss histories over the duration of the run.
* **Visualization:** Plots the training and validation loss curves over time to verify stable convergence and ensure the models do not suffer from overfitting.

### 3. `generate.ipynb` (Inference & Evaluation)

This notebook evaluates the fully trained "XL" model on both quantitative metrics and qualitative generation tasks.

* **Quantitative Evaluation:** Calculates the final Cross-Entropy Loss and Perplexity over the completely held-out test set to measure the model's fundamental sequence modeling performance.
* **Generative Sampling:** Performs unconditional and conditional SVG generation using temperature scaling and Top-K filtering.
* **Structural Evaluation:** Evaluates the generated outputs based on strict formatting rules, testing for *XML validity* (via `lxml`) and *renderability* (via `CairoSVG`).

## Requirements and Setup

To run these notebooks, ensure you have the following dependencies installed (and preferably a CUDA-capable GPU):

```bash
torch torchvision torchaudio transformers datasets tokenizers matplotlib numpy cairosvg lxml mup
```

There is a requirements.txt file provided for convienence:

```bash
pip install -r requirements.txt
```

*Note: The datasets are cached locally. You may need to configure your `HF_HOME` environment variable as seen in the first cell of `main.ipynb` to direct the Hugging Face cache to your preferred directory.*

## Usage

1. Open and run `main.ipynb` to download the data, build the tokenizer, and run the initial hyperparameter sweeps.
2. Open and run `train.ipynb` to train the full-scale models using the optimal learning rate discovered in step 1.
3. Open and run `generate.ipynb` to load the saved model weights, calculate test perplexity, and generate visual SVG samples.
