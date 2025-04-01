# Sentiment Analysis Using LSTM and Word2Vec

## Overview
This project implements a sentiment analysis model using LSTM (Long Short-Term Memory) and Word2Vec word embeddings. It classifies text as either **positive** or **negative** based on a small dataset of sample sentences.

## Features
- Uses **NLTK** for tokenization.
- Implements **Word2Vec** for word embeddings.
- Utilizes **LSTM** in a **Keras** Sequential model.
- Trains on a small dataset and performs sentiment predictions.

## Requirements
Make sure you have the following dependencies installed before running the code:

```bash
pip install numpy tensorflow nltk gensim scikit-learn
```

## Dataset
The dataset consists of six labeled sentences:
- **Positive (1)**: Expressions of satisfaction or enjoyment.
- **Negative (0)**: Expressions of dissatisfaction or disappointment.

## Workflow
1. **Preprocess Data**: Tokenizes sentences using NLTK.
2. **Train Word2Vec Model**: Converts words into 100-dimensional vectors.
3. **Tokenization & Padding**: Converts text into sequences and pads them.
4. **Prepare Embedding Matrix**: Maps words to their Word2Vec embeddings.
5. **Train LSTM Model**: Uses the embeddings to train an LSTM-based classifier.
6. **Prediction**: Tests sentiment on a sample sentence.

## Model Architecture
- **Embedding Layer**: Uses pretrained Word2Vec embeddings.
- **LSTM Layer**: Captures sequential dependencies in text.
- **Dense Layer**: Outputs a probability score using sigmoid activation.

## Training
The model is trained using binary cross-entropy loss and Adam optimizer for **10 epochs** with a batch size of **2**.

## Example Prediction
After training, the model evaluates a test sentence:

```python
Test Sentence: "I really love this product"
Sentiment Score: 0.8745 (Positive)
```

## Running the Code
Simply execute the script:

```bash
python sentiment_analysis.py
```

## Future Enhancements
- Expand the dataset for better generalization.
- Implement data augmentation.
- Fine-tune hyperparameters for better accuracy.

## Author
Developed by [Ibrahim Shedoh].

