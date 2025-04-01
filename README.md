# Bangla-News-Classification-with-BERT
Bangla News Classification: A Comparative Analysis of Transformer and Sequential Architectures
This repository contains code and resources for classifying Bangla news articles into different categories using various machine learning models. The project focuses on comparing the performance of transformer-based architectures (BERT, DistilBERT) with traditional sequential models (BiLSTM, CNN, GRU).

Project Overview
The goal of this project is to build and evaluate different models for Bangla news classification, providing a comprehensive analysis of their strengths and weaknesses. We leverage pre-trained language models such as BERT and DistilBERT, along with custom-built sequential models, to achieve this.

Key Features
Comparative Analysis:
Directly compares the performance of transformer-based models with sequential models on Bangla news classification.

Hyperparameter Tuning:
Utilizes Optuna for automated hyperparameter optimization to determine the best settings for each model.

5-Fold Cross-Validation:
Implements 5-fold cross-validation to ensure robust and reliable model evaluation.

Detailed Metrics:
Reports a variety of classification metrics including accuracy, precision, recall, and F1-score.

Visualization Tools:
Provides confusion matrices and detailed classification reports to understand model behavior and error patterns.

Bangla Text Preprocessing:
Includes functions for basic Bangla text cleaning and tokenization.

Dataset
The project uses a Bangla news dataset (https://www.kaggle.com/datasets/durjoychandrapaul/over-11500-bangla-news-for-nlp) containing news articles and their corresponding categories. The dataset is expected to have the following columns:

title: Title of the news article.

published_date: Date of publication.

reporter: Name of the reporter 

category: Category of the news article.

url: URL of the news article.

content: Text content of the news article.

Data Preprocessing
Label Encoding:
Categories are encoded into numerical labels using sklearn.preprocessing.LabelEncoder.

Text Cleaning:
Basic cleaning is applied to the text content, including lowercasing, punctuation removal, and preserving Bangla character ranges.

Tokenization:
Text is tokenized using a simple whitespace-based tokenizer for custom models, and pre-trained tokenizers for transformer-based models.

Models
The following models are implemented and compared in this repository:

BERT (bert-base-multilingual-cased):
A pre-trained transformer model fine-tuned for Bangla news classification.

DistilBERT (distilbert-base-multilingual-cased):
A lighter, faster version of BERT, also fine-tuned for the classification task.

BiLSTM:
A bidirectional LSTM model incorporating an embedding layer, LSTM layers, and a fully connected output layer.

CNN:
A convolutional neural network featuring an embedding layer, convolutional layers, max-pooling, and a fully connected output layer.

GRU:
A gated recurrent unit model with an embedding layer, GRU layers, and a fully connected output layer.

Observations:
Highlight interesting patterns such as which model better captures the nuances in Bangla news text or how different architectures handle various news categories.

Future Work:
Our future work will focus on several promising avenues. We 
plan to implement advanced attention mechanisms and ensemble 
approaches to better differentiate between overlapping categories 
and enhance classification robustness. Dynamic strategies that 
adjust dropout or other regularization parameters during training 
will also be explored to improve model generalizability. In addition, 
the integration of multimodal data represents an exciting direction for further research. By combining textual information with other modalities—such as images, audio, or video—we aim to capture richer contextual cues and semantic details that can significantly 
improve classification performance in real-world applications. 
Multimodal approaches could facilitate a more comprehensive 
understanding of Bangla news content, enabling models to leverage 
visual and auditory signals alongside text to resolve ambiguities and enhance decision-making in complex scenarios. 

Contributing
Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

License
This project is licensed under the MIT License.

Acknowledgements
I would like to express my deepest gratitude to my Deep 
Learning course instructor, Md. Mynoddin, for his 
unwavering support and insightful guidance throughout my 
academic journey. His profound expertise and dedication 
have been instrumental in deepening my understanding of 
complex concepts, and his encouragement has continually 
inspired me to strive for excellence. I am truly fortunate to 
have had the opportunity to learn under his mentorship.
