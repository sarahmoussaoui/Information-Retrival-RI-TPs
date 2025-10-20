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
# 5. INDEX BUILDING (Document-term & Inverted index)
# ------------------------------------------------------------
doc_index = defaultdict(list)
inverted_index = defaultdict(list)

for doc_name, text in docs.items():
    tokens = tokenize_regex(text)
    tokens = remove_stopwords(tokens)
    stems = stem_porter(tokens)  # or use stem_LABcaster 

    # Create descriptor: <Document number> <Term>
    for term in stems:
        doc_index[doc_name].append(term)
        inverted_index[term].append(doc_name)

# ------------------------------------------------------------
# SAVE DESCRIPTOR FILES
# ------------------------------------------------------------
with open("LAB0+1/results/descriptor.txt", "w") as f:
    for doc, terms in doc_index.items():
        for term in terms:
            f.write(f"{doc}\t{term}\n")

# ------------------------------------------------------------
# SAVE INVERTED INDEX FILE
# ------------------------------------------------------------
with open("LAB0+1/results/inverted_index.txt", "w") as f:
    for term, docs_ in inverted_index.items():
        for doc in set(docs_):
            f.write(f"{term}\t{doc}\n")

print("Descriptor and Inverted index files created.")

# ------------------------------------------------------------
# 6. TERM FREQUENCY & WEIGHTING (TF-IDF)
'''A term that appears many times in one document → high TF

A term that appears in every document → low IDF

The product highlights terms that are frequent but distinctive'''
# ------------------------------------------------------------
# Prepare corpus for sklearn TF-IDF
corpus = [text for text in docs.values()]
vectorizer = TfidfVectorizer(
    stop_words='english',
    tokenizer=tokenize_and_stem,       # use your regex tokenizer
    token_pattern=None              # disable default token_pattern
)
tfidf_matrix = vectorizer.fit_transform(corpus) # computes values for every term
terms = vectorizer.get_feature_names_out() # all unique terms

# Display TF-IDF results
df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=terms, index=docs.keys()) # each cell [i, j] = TF-IDF weight of term j in document i.
print("\nTF-IDF matrix:\n", df_tfidf.round(3))

# ------------------------------------------------------------
# SAVE UPDATED DESCRIPTOR & INVERTED INDEX WITH FREQUENCY & WEIGHT
# ------------------------------------------------------------
with open("LAB0+1/results/descriptor_weighted.txt", "w") as f:
    for i, doc in enumerate(docs.keys()):
        for j, term in enumerate(terms):
            freq = tfidf_matrix[i, j]
            if freq > 0:
                f.write(f"{doc}\t{term}\t{freq:.3f}\n")

with open("LAB0+1/results/inverted_index_weighted.txt", "w") as f:
    for j, term in enumerate(terms):
        for i, doc in enumerate(docs.keys()):
            freq = tfidf_matrix[i, j]
            if freq > 0:
                f.write(f"{term}\t{doc}\t{freq:.3f}\n")

print("Weighted descriptor and inverted index created successfully.")



