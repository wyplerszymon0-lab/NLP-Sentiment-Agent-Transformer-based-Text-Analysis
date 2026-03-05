# NLP Sentiment Agent: Transformer-based Text Analysis 

This project demonstrates the implementation of a Natural Language Processing (NLP) agent using the **Transformer** architecture. It leverages the **DistilBERT** model to perform high-accuracy sentiment analysis.

##  Technical Highlights
- **Architecture**: Based on **BERT (Bidirectional Encoder Representations from Transformers)**, optimized for inference speed.
- **Library**: Utilizes `Hugging Face Transformers` with a **PyTorch** backend.
- **Inference**: Implements a reusable `SentimentAgent` class for modular integration into larger systems.
- **Model**: `distilbert-base-uncased-finetuned-sst-2-english` – a gold standard for text classification tasks.

##  Why Transformers?
Unlike traditional RNNs or LSTMs, Transformers use **Self-Attention mechanisms** to process text in parallel, making them the backbone of modern AI like GPT-4 and Claude.

##  Quick Start
```bash
pip install transformers torch
python nlp_sentiment_agent.py
