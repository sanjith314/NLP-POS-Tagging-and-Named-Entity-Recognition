NLP POS Tagging and Named Entity Recognition

A Natural Language Processing project for Part-of-Speech (POS) Tagging and Named Entity Recognition (NER) using Python and Stanford NLP tools. This project implements a statistical POS tagger with Viterbi decoding and integrates Stanford Named Entity Recognizer to extract entities from movie summaries.

Overview:
This project demonstrates core NLP techniques including:
Preprocessing and training on the Penn Treebank corpus using NLTK
Building a Hidden Markov Model (HMM) POS tagger
Decoding POS tags using the Viterbi algorithm
Performing Named Entity Recognition using Stanford NER
Extracting entities such as PERSON, LOCATION, ORGANIZATION, and MONEY

Features

POS Tagging:
Uses the Penn Treebank corpus from NLTK
Computes transition and emission probabilities
Implements the Viterbi decoding algorithm
Evaluates tagging accuracy on test data

Named Entity Recognition:
Uses Stanford NER through NLTK’s StanfordNERTagger
Extracts named entities from movie plot summaries
Filters out non-entity tokens labeled as 'O'
Supports a 7-class English NER model

Prerequisites:
Python 3.7 or higher
Java JDK/JRE 1.8 or higher (required for Stanford NER)

To run Named Entity Recognition:
python ner.py
This will load movie summaries from dataset.csv and print extracted named entities for each movie.
For POS tagging, complete the implementation in parser.py and run the corresponding script to evaluate POS tagging performance.

Notes:
Stanford NER is a Java-based CRF model trained on English text.
Only the JAR file and required classifier model are necessary; other Stanford NER files can be removed to reduce repository size.

License:
This project is intended for academic and educational purposes.