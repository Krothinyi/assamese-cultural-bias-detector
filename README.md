# Assamese Cultural Bias Detector

**A multilingual cultural bias detection system using Qwen2.5-0.5B-Instruct, LoRA fine-tuning, and Gradio.**

This project investigates the identification of cultural bias in social media comments relating to Assamese identity, language, food, and appearance.

## Overview

The application uses a fine-tuned language model to classify Assamese-related YouTube comments.

For each input comment, the model predicts:

- **Bias classification:** Biased (1) or Non-biased (0)
- **Cultural category:** Language, Food, or Appearance

The application provides an interactive Gradio interface for analyzing comments.

The current release uses **Version 3 (V3)** of the fine-tuned classifier.

## Technology Stack

- **Programming language:** Python
- **Base model:** Qwen/Qwen2.5-0.5B-Instruct
- **Fine-tuning:** Low-Rank Adaptation (LoRA)
- **Deep learning:** PyTorch
- **LLM framework:** Hugging Face Transformers
- **Parameter-efficient fine-tuning:** PEFT
- **Web interface:** Gradio

## Model Architecture

The system combines the pretrained Qwen2.5-0.5B-Instruct model with a fine-tuned LoRA adapter.

**Inference pipeline:**

User Comment → Tokenization → Qwen2.5 + V3 LoRA → JSON Prediction → Gradio Interface

The output contains a binary bias label and a cultural category.

The base model is loaded from Hugging Face, while the trained V3 adapter is stored locally in the project.

## Dataset

The V3 experiment uses human-labeled Assamese-related YouTube comments.

| Dataset | Number of comments |
|---|---:|
| Training | 200 |
| Validation | 44 |
| Reused diagnostic | 37 |

**Training class distribution:**

- Non-biased: 180
- Biased: 20

Training and validation videos are disjoint to reduce video-level information leakage.

The diagnostic dataset was reused during model development and is not an untouched final test set.

## Model Performance — V3

| Metric | Validation | Reused Diagnostic |
|---|---:|---:|
| Accuracy | 81.82% | 67.57% |
| Macro F1 | 0.6944 | 0.6119 |
| Biased Precision | 0.5000 | 0.7143 |
| Biased Recall | 0.5000 | 0.3333 |
| Biased F1 | 0.5000 | 0.4545 |
| Category Accuracy | 97.73% | 81.08% |

These results are based on small evaluation datasets and should not be interpreted as established real-world performance.

## Installation

Clone the repository:

```bash
git clone https://github.com/Krothinyi/assamese-cultural-bias-detector.git
cd assamese-cultural-bias-detector
```

Install the required dependencies in a suitable Python environment:

```bash
pip install -r requirements.txt
```

Launch the application:

```bash
python app.py
```

Open the local URL printed in the terminal to access the Gradio interface.

The application downloads the pretrained Qwen model from Hugging Face when it is not already cached.

**Note:** The prototype has been tested with Python 3.12 and its existing project environment. A compatible CUDA GPU with BF16 support is recommended. CPU inference is supported but may be substantially slower.

## Project Structure

```text
assamese-cultural-bias-detector/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── assets/
└── data_/
    └── qwen_lora_adapter_v3_video_disjoint/
        ├── adapter_model.safetensors
        ├── adapter_config.json
        ├── pilot_metadata.json
        └── tokenizer files
```

## Research Limitations

- The V3 training dataset is relatively small and imbalanced.
- Bias detection recall remains limited.
- Performance across English, Assamese script, Romanized Assamese, and Hindi has not been independently established.
- The classifier may struggle with implicit stereotypes, sarcasm, contextual meaning, and code-mixed comments.
- The model assigns one of three cultural categories and cannot reliably identify irrelevant or out-of-scope comments.
- The system is a research prototype, not a production-ready content moderation tool.

## Future Work — Version 5

Ongoing research investigates improvements to cultural bias detection through:

- Expanded human-annotated datasets
- Improved handling of class imbalance
- Alternative fine-tuning strategies
- Cross-lingual robustness evaluation
- Better detection of subtle cultural bias

Version 5 experiments have not yet demonstrated a consistent improvement over V3. Therefore, **V3 remains the deployed application model**.

## Project Status

**Current release:** V3 research prototype

**Ongoing development:** Expanded dataset and model improvement experiments

This project is part of ongoing academic research on cultural bias identification in language models.