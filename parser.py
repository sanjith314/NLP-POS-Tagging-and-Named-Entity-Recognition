# CS421: Natural Language Processing
# University of Illinois at Chicago
# Spring 2026
# Assignment 1
#
# Do not rename/delete any functions or global variables provided in this template and write your solution
# in the specified sections. Use the main function to test your code when running it from a terminal.
# Avoid writing that code in the global scope; however, you should write additional functions/classes
# as needed in the global scope. These templates may also contain important information and/or examples
# in comments so please read them carefully. If you want to use external packages not specified in the
# assignment then you need prior approval from course staff.
#
# This code will be graded automatically using Gradescope.
# =========================================================================================================
import nltk
from nltk.corpus import treebank
import numpy as np

import random

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('treebank')

# Function: get_treebank_data
# Input: None
# Returns: Tuple (train_sents, test_sents)
#
# This function fetches tagged sentences from the NLTK Treebank corpus, calculates an index for an 80-20 train-test split,
# then splits the data into training and testing sets accordingly.

def get_treebank_data():
    tagged_sents = treebank.tagged_sents()

    split_idx = int(0.8 * len(tagged_sents))

    train_sents = tagged_sents[:split_idx]
    test_sents = tagged_sents[split_idx:]

    vocab = set()
    tagset = set()

    for sent in tagged_sents:
        for word, tag in sent:
            vocab.add(word)
            tagset.add(tag)

    return train_sents, test_sents, vocab, tagset


# Function: compute_tag_trans_probs
# Input: train_data (list of tagged sentences)
# Returns: Dictionary A of tag transition probabilities
#
# Iterates over training data to compute the probability of tag bigrams (transitions from one tag to another).

from collections import defaultdict

def compute_tag_trans_probs(train_sents):
    tag_bigram_counts = defaultdict(lambda: defaultdict(int))
    tag_counts = defaultdict(int)

    for sent in train_sents:
        prev_tag = "<START>"
        tag_counts[prev_tag] += 1

        for _, tag in sent:
            tag_bigram_counts[prev_tag][tag] += 1
            tag_counts[tag] += 1
            prev_tag = tag

    trans_probs = defaultdict(dict)

    for prev_tag in tag_bigram_counts:
        for curr_tag in tag_bigram_counts[prev_tag]:
            trans_probs[prev_tag][curr_tag] = (
                tag_bigram_counts[prev_tag][curr_tag] /
                tag_counts[prev_tag]
            )

    return trans_probs


# Function: compute_emission_probs
# Input: train_data (list of tagged sentences)
# Returns: Dictionary B of tag-to-word emission probabilities
#
# Iterates through each sentence in the training data to count occurrences of each tag emitting a specific word, then calculates probabilities.

def compute_emission_probs(train_sents):
    word_tag_counts = defaultdict(lambda: defaultdict(int))
    tag_counts = defaultdict(int)

    for sent in train_sents:
        for word, tag in sent:
            word_tag_counts[tag][word] += 1
            tag_counts[tag] += 1

    emit_probs = defaultdict(dict)

    for tag in word_tag_counts:
        for word in word_tag_counts[tag]:
            emit_probs[tag][word] = (
                word_tag_counts[tag][word] /
                tag_counts[tag]
            )

    return emit_probs


# Function: viterbi_algorithm
# Input: words (list of words that have to be tagged), A (transition probabilities), B (emission probabilities)
# Returns: List (the most likely sequence of tags for the input words)
#
# Implements the Viterbi algorithm to determine the most likely tag path for a given sequence of words, using given transition and emission probabilities.

def viterbi_algorithm(words, A, B):
    states = list(B.keys())
    Vit = [{}]
    path = {}
    for state in states:
        Vit[0][state] = B.get(state, {}).get(words[0], 0.0001)
        path[state] = [state]

    for t in range(1, len(words)):
        pass

    return None

# Function: evaluate_pos_tagger
# Input: test_data (tagged sentences for testing), A (transition probabilities), B (emission probabilities)
# Returns: Float (accuracy of the POS tagger on the test data)
#
# Evaluates the POS tagger's accuracy on a test set by comparing predicted tags to actual tags and calculating the percentage of correct predictions.

def evaluate_pos_tagger(test_data, A, B):
    correct = 0
    total = 0
    accuracy = None
    return accuracy


# Use this main function to test your code. Sample code is provided to assist with the assignment;
# feel free to change/remove it. Some of the provided sample code will help you in answering
# questions, but it won't work correctly until all functions have been implemented.
if __name__ == "__main__":
    # Main function to train and evaluate the POS tagger.


    train_data, test_data = get_treebank_data()
    A = compute_tag_trans_probs(train_data)
    B = compute_emission_probs(train_data)

    # Print specific probabilities
    print(f"P(VB -> DT): {A['VB'].get('DT', 0):.4f}")  # Expected Probability should be around 0.2296
    print(f"P(DT -> 'the'): {B['DT'].get('the', 0):.4f}")  # Expected Probability should be around 0.4986
    
    # Evaluate the model's accuracy
    accuracy = evaluate_pos_tagger(test_data, A, B)
    print(f"Accuracy of the HMM-based POS Tagger: {accuracy:.4f}") ## Expected accuracy around 0.8743



