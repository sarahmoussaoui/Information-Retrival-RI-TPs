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
N = len(docs)
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
import pandas as pd

import math
from collections import Counter

# ------------------------------------------------------------
# 6. MANUAL TF, IDF, AND TF-IDF COMPUTATION
# ------------------------------------------------------------

# ------------------------------------------------------------
# TOKENIZE AND COUNT FREQUENCIES
# ------------------------------------------------------------
tokenized_docs = {doc: tokenize_and_stem(text) for doc, text in docs.items()}

# 1️⃣ Term frequencies per document
term_freqs = {doc: Counter(tokens) for doc, tokens in tokenized_docs.items()}

# 2️⃣ Document frequencies (dfₜ)
df = Counter()
for tokens in tokenized_docs.values():
    for term in set(tokens):
        df[term] += 1

# ------------------------------------------------------------
# COMPUTE CUSTOM TF–IDF
# ------------------------------------------------------------
tfidf = {}
for doc, freqs in term_freqs.items():
    tfidf[doc] = {}
    max_freq = max(freqs.values())  # max frequency in this document
    for term, f_td in freqs.items():
        tf = f_td / max_freq
        idf = math.log10((N / df[term]) + 1)
        tfidf[doc][term] = tf * idf

# ------------------------------------------------------------
# SAVE RESULTS
# ------------------------------------------------------------
os.makedirs("LAB2/results", exist_ok=True)

# Descriptor: Document → Term → Frequency → TF–IDF
with open("LAB2/results/descriptor_weighted.txt", "w", encoding="utf-8") as f:
    for doc, terms in term_freqs.items():
        doc_id = os.path.splitext(doc)[0]
        for term, freq in terms.items():
            f.write(f"{doc_id}\t{term}\t{freq}\t{tfidf[doc][term]:.6f}\n")

# Inverted Index: Term → Document → Frequency → TF–IDF
with open("LAB2/results/inverted_index_weighted.txt", "w", encoding="utf-8") as f:
    for term in sorted(df.keys()):
        for doc in docs.keys():
            if term in term_freqs[doc]:
                freq = term_freqs[doc][term]
                f.write(f"{term}\t{os.path.splitext(doc)[0]}\t{freq}\t{tfidf[doc][term]:.6f}\n")

print("✅ Custom TF–IDF (normalized by max freq, log10) computed successfully.")
