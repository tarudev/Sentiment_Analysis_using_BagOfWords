# Sentiment Analysis using BagOfWords
A tool for Raw Text processing and deciphering the Sentiment as positive or negative based on training ML (Machine Learning) models on a dataset of reviews. The Bag Of Words model is coupled with the NLP (Natural Language Processing) method for text preprocessing, tokenisation and vectorisation to predict the sentiment as positive or negative for a certain review.

For the purpose of sentiment deciphering, we must tackle raw text which is unconventional as ML algorithms process only numeric data. So, we use the methods of Natural Language Processing to assess the raw text into a numeric form. 
The usual methods of rectifying the null entries, missing entries and removing unrelated data columns are employed to smoothen and speed up the process of feature extraction.
A fair bit of preprocessing is performed on the text such that it can be fed to the ML models:

1) Removing the regular expressions:
    * Remove email addresses
    * Remove URLs
    * Remove non-letters
    * Remove numbers
2) Convert to lower case, split into individual words
3) Gather the list of stopwords in English Language
4) Remove stop words and stemming the remaining words
5) Join the tokens back into one string separated by space and return the result.

For feature extraction, we are using **Bag Of Words model**:

1) It is a simple method of text manipulation and feature extraction.
2) It detects the presence of a word in the review/text.
3) The occurence of the word is the feature extracted and each word is either assigned 0 or 1 as per its presence in the text
4) It does not concerns itself with the order of the presence of the word. Thus, it remains semantic.

A better approach can be considering n-grams, TF-IDF, etc.

After tokenisation and vectorisation as per Bag Of Words, we obtain sparse matrix representation of the text/input data.
This is fed to the ML models of:
1. Naive Bayes Theorem
2. Random Forest Classifier
3. Decision Tree

https://machinelearningmastery.com/gentle-introduction-bag-words-model/
