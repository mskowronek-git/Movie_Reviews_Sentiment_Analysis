# Movie_Reviews_Sentiment_Analysis

##Overview
This project focuses on sentiment analysis using the Large Movie Review Dataset, employing a variety of machine learning techniques. It is part of a machine learning course project, where the goal is to clean the data, train, test, and evaluate different models for sentiment analysis. Both traditional and modern models are explored for training and testing. Contributions and suggestions for improvements are welcome.

##Dataset
The dataset consists of 50,000 IMDB movie reviews, split into 25,000 training and 25,000 testing reviews. The sentiment labels (positive and negative) are balanced across the dataset.

The data is preprocessed by performing the following steps:

- Removing stopwords
- Converting to lowercase
- Removing punctuation
- Tokenizing

##Models
The following models were implemented and evaluated:
- Traditional ML Models
- Logistic Regression
- Decision Tree
- AdaBoost with Decision Tree base estimator
- Neural Network Models
- Simple feedforward network
- Deep network with dropout and regularization

##Word Embedding Models:
- TF-IDF vectors
- FastText word embeddings trained in unsupervised mode

##Training
Models are trained on 80% of the dataset, with 10% used for hyperparameter tuning.

The TensorFlow-based models use the Adam optimizer and sparse categorical cross-entropy loss function. These models are trained for 30 epochs with a batch size of 32.

The FastText model is trained on the training set sentences using supervised learning.

##Evaluation
All models are evaluated on the remaining 10% test set, with performance measured by accuracy and F1 score.

Classification reports are generated to show precision, recall, and F1 score for each sentiment class.

##Results
The deep neural network model performs the best, achieving a test accuracy of 63%. Logistic regression, decision tree, and FastText models also show reasonable performance.

In general, neural network models outperform traditional machine learning models. Word embedding models, such as FastText, outperform TF-IDF vectors.

There is potential for further improvement by incorporating advanced architectures, pretrained embeddings, and more regularization techniques.
