# # LAB 0 

# %%
# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

# LAB0
# Text = """ In 2025, Garfild the cat ate a whole plate of lasagna. """

# #terms = Text.split()
# # print("extracted terms:\n",terms)

# import nltk
# from nltk.tokenize import RegexpTokenizer

# ExpReg = RegexpTokenizer(r'\w+')
# terms = ExpReg.tokenize(Text)
# print(terms)

# # LAB 1


import os
import re
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

nltk.download('stopwords')

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
folder = "Collection"   # Folder containing D1.txt, D2.txt, etc.
stop_words = set(stopwords.words('english'))


# ------------------------------------------------------------
# 1. READ COLLECTION
# ------------------------------------------------------------
docs = {}
for file in os.listdir(folder):
    if file.endswith(".txt"):
        with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
            docs[file] = f.read()

print("Documents read:", list(docs.keys()))

# ------------------------------------------------------------
# 2. TOKENIZATION
# ------------------------------------------------------------

def tokenize_split(text):
    """Tokenization using simple split()"""
    return text.lower().split()

def tokenize_regex(text):
    """Tokenization using regular expressions (keeping words and abbreviations)"""
    tokenizer = RegexpTokenizer(
        r'(?:[A-Za-z]\.)+'            # abbreviations like U.S.A.
        r'|(?:[A-Za-z]+[\-@]\d+(?:\.\d+)?)'  # words with - or @ and numbers like A-123 or A@123
        r'|\d+(?:[.,-]\d+)*%?'        # numbers, decimals, percentages
        r'|[A-Za-z]+'                 # regular words
    )
    return tokenizer.tokenize(text.lower())

def tokenize_and_stem(text):
    tokens = tokenize_regex(text)
    tokens = remove_stopwords(tokens)
    stems = stem_porter(tokens)
    return stems


# ------------------------------------------------------------
# 3. STOPWORD REMOVAL
# ------------------------------------------------------------
def remove_stopwords(tokens):
    return [t for t in tokens if t not in stop_words]

# ------------------------------------------------------------
# 4. NORMALIZATION (STEMMING)
# ------------------------------------------------------------
porter = PorterStemmer()

def stem_porter(tokens):
    return [porter.stem(t) for t in tokens]


# ------------------------------------------------------------
# 6. TERM FREQUENCY & WEIGHTING (TF-IDF)
# ------------------------------------------------------------
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd

# Prepare corpus
corpus = [text for text in docs.values()]
doc_names = list(docs.keys())

# ---- 1. Compute raw term frequencies (counts) ----
count_vectorizer = CountVectorizer(
    stop_words='english',
    tokenizer=tokenize_and_stem,
    token_pattern=None
)
term_freq_matrix = count_vectorizer.fit_transform(corpus)
terms = count_vectorizer.get_feature_names_out()

# ---- 2. Compute TF-IDF weights ----
tfidf_vectorizer = TfidfVectorizer(
    stop_words='english',
    tokenizer=tokenize_and_stem,
    token_pattern=None
)
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
terms_tfidf = tfidf_vectorizer.get_feature_names_out()

# Ensure terms match (they should if tokenization is identical)
assert list(terms) == list(terms_tfidf), "Terms mismatch between TF and TF-IDF!"

# ------------------------------------------------------------
# 7. SAVE WEIGHTED DESCRIPTOR FILE
# Format: <Document> <Term> <Frequency> <TF-IDF>
# ------------------------------------------------------------
with open("LAB0+1/results/descriptor_weighted.txt", "w") as f:
    for i, doc in enumerate(doc_names):
        # remove .txt extension if present
        doc_id = os.path.splitext(doc)[0]
        for j, term in enumerate(terms):
            freq = term_freq_matrix[i, j]
            tfidf = tfidf_matrix[i, j]
            if freq > 0:
                f.write(f"{doc_id}\t{term}\t{freq}\t{tfidf:.4f}\n")

# ------------------------------------------------------------
# 8. SAVE WEIGHTED INVERTED INDEX FILE
# Format: <Term> <Document> <Frequency> <TF-IDF>
# ------------------------------------------------------------
with open("LAB0+1/results/inverted_index_weighted.txt", "w") as f:
    for j, term in enumerate(terms):
        for i, doc in enumerate(doc_names):
            # remove .txt extension
            doc_id = os.path.splitext(doc)[0]
            freq = term_freq_matrix[i, j]
            tfidf = tfidf_matrix[i, j]
            if freq > 0:
                f.write(f"{term}\t{doc_id}\t{freq}\t{tfidf:.4f}\n")

print("✅ Weighted descriptor and inverted index created successfully.")
