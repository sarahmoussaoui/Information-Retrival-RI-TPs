import os
import math
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import defaultdict

# Télécharger stopwords si nécessaire (silencieux)
nltk.download('stopwords', quiet=True)

# Paths
DOC_TERMS_PATH = "LAB4/results/inverted_index_weighted.txt"
COLLECTION_DIR = "Collection"

# Tokenizer / stopwords / stemmer (same pipeline as other scripts)
ExpReg = RegexpTokenizer(
    r'\(|\)|(?:[A-zA-z]\.)+'                # ( ) and abbreviations like U.S.A.
    r'|[A-za-z]+[\-@]\d+(?:\.\d+)?'         # tokens with - or @ followed by digits
    r'|\d+(?:[\.\,\-]\d+)*%?'               # numbers, percents
    r'|[A-Za-z]+'                           # words
)
StopWords = set(stopwords.words('english'))
Porter = PorterStemmer()


def preprocess_text(text):
    """Tokenize, remove stopwords, apply Porter stemming"""
    tokens = ExpReg.tokenize(text)
    tokens_nostop = [t for t in tokens if t.lower() not in StopWords and t.strip() != '']
    stems = [Porter.stem(t.lower()) for t in tokens_nostop]
    return stems


def load_document_terms(path=DOC_TERMS_PATH):
    """Load term frequencies from inverted_index.txt"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} introuvable — exécute Lab1 pour générer ce fichier.")
    
    # Structure: doc_id -> {term: frequency}
    # inverted_index.txt format: term doc_id freq tfidf
    doc_terms = defaultdict(dict)
    docs_set = set()
    vocabulary = set()
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            term, doc_id = parts[0], parts[1]
            tf = int(parts[2])  # term frequency
            
            docs_set.add(doc_id)
            doc_terms[doc_id][term] = tf
            vocabulary.add(term)
    
    docs = sorted(docs_set)
    return dict(doc_terms), docs, sorted(vocabulary)


def compute_document_lengths(doc_terms):
    """Compute Nd (document length) for each document"""
    doc_lengths = {}
    for doc_id, terms in doc_terms.items():
        doc_lengths[doc_id] = sum(terms.values())
    return doc_lengths


def compute_collection_frequencies(doc_terms, vocabulary):
    """Compute cf(w,C) - collection frequency for each term"""
    cf = defaultdict(int)
    for doc_id, terms in doc_terms.items():
        for term, freq in terms.items():
            cf[term] += freq
    return dict(cf)


def good_turing_smoothing(query_terms, doc_terms, doc_lengths, vocabulary):
    """
    Model 3: Good-Turing smoothing (approximate implementation)

    We use the simple Good-Turing adjusted counts:
      c* = (c+1) * N_{c+1} / N_c   if N_{c+1} > 0
      otherwise fall back to c

    For unseen terms we distribute the mass N1/ Nd equally among unseen types.
    This is an approximate but practical variant for manual verification.
    """
    scores = {}
    V = len(vocabulary)

    for doc_id in doc_terms.keys():
        Nd = doc_lengths[doc_id]

        # frequency-of-frequencies for this document
        freq_of_freq = defaultdict(int)
        for t, c in doc_terms[doc_id].items():
            freq_of_freq[c] += 1

        N1 = freq_of_freq.get(1, 0)
        observed_types = set(doc_terms[doc_id].keys())
        V0 = max(0, V - len(observed_types))

        # compute adjusted counts
        c_star = {}
        for t, c in doc_terms[doc_id].items():
            Nc = freq_of_freq.get(c, 0)
            Ncp1 = freq_of_freq.get(c+1, 0)
            if Nc > 0 and Ncp1 > 0:
                cstar = (c+1) * (Ncp1 / Nc)
            else:
                cstar = c
            c_star[t] = cstar

        # sum of adjusted counts for observed types
        S = sum(c_star.values())

        # assign probability mass for unseen types
        if V0 > 0:
            p0 = (N1 / Nd) / V0 if Nd > 0 else 0.0
        else:
            p0 = 0.0

        log_likelihood = 0.0
        for term in query_terms:
            if term in doc_terms[doc_id]:
                prob = c_star[term] / Nd if Nd > 0 else 0.0
            else:
                prob = p0

            # avoid zero or negative probabilities
            if prob <= 0:
                log_likelihood = -float('inf')
                break
            log_likelihood += math.log10(prob)

        scores[doc_id] = log_likelihood

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked


def jelinek_mercer_smoothing(query_terms, doc_terms, doc_lengths, vocabulary, collection_cf, lamb=0.5):
    """
    Model 4: Jelinek-Mercer smoothing (linear interpolation)

    P(w|d) = lambda * P_MLE(w|d) + (1-lambda) * P_MLE(w|C)
           = lambda * tf(w,d)/Nd + (1-lambda) * cf(w)/|C|
    where |C| is total tokens in collection.
    """
    scores = {}
    total_collection_terms = sum(collection_cf.values())

    for doc_id in doc_terms.keys():
        Nd = doc_lengths[doc_id]
        log_likelihood = 0.0
        for term in query_terms:
            tf_wd = doc_terms[doc_id].get(term, 0)
            p_doc = (tf_wd / Nd) if Nd > 0 else 0.0
            p_coll = (collection_cf.get(term, 0) / total_collection_terms) if total_collection_terms > 0 else 0.0
            p = lamb * p_doc + (1 - lamb) * p_coll
            if p <= 0:
                log_likelihood = -float('inf')
                break
            log_likelihood += math.log10(p)
        scores[doc_id] = log_likelihood

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked


def dirichlet_smoothing(query_terms, doc_terms, doc_lengths, vocabulary, collection_cf, mu=None):
    """
    Model 5: Dirichlet smoothing

    P(w|d) = (tf(w,d) + mu * P_MLE(w|C)) / (|d| + mu)
    where P_MLE(w|C) = cf(w)/|C|
    
    If mu is not provided, it is computed as:
    μ = 0.3 × N_avg
    where N_avg is the average document length in the collection
    """
    scores = {}
    total_collection_terms = sum(collection_cf.values())
    
    # Compute μ if not provided
    if mu is None:
        N_avg = sum(doc_lengths.values()) / len(doc_lengths)
        mu = 0.3 * N_avg

    for doc_id in doc_terms.keys():
        Nd = doc_lengths[doc_id]
        log_likelihood = 0.0
        for term in query_terms:
            tf_wd = doc_terms[doc_id].get(term, 0)
            p_coll = (collection_cf.get(term, 0) / total_collection_terms) if total_collection_terms > 0 else 0.0
            p = (tf_wd + mu * p_coll) / (Nd + mu) if (Nd + mu) > 0 else 0.0
            if p <= 0:
                log_likelihood = -float('inf')
                break
            log_likelihood += math.log10(p)
        scores[doc_id] = log_likelihood

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked, mu


def mle_unsmoothed(query_terms, doc_terms, doc_lengths, vocabulary):
    """
    Model 1: Unsmoothed Language Model (MLE)
    
    P(w|d) = tf(w,d) / Nd
    
    Score(q,d) = Π P(w|d) for w in q
    Log-likelihood: Σ log P(w|d) for w in q
    
    Warning: If a query term is not in document, P(w|d) = 0 → score = -∞
    """
    scores = {}
    
    for doc_id in doc_terms.keys():
        log_likelihood = 0.0
        Nd = doc_lengths[doc_id]
        
        for term in query_terms:
            tf_wd = doc_terms[doc_id].get(term, 0)
            
            if tf_wd == 0:
                # Term not in document → P(w|d) = 0
                # log(0) = -∞, so we assign a very large negative value
                log_likelihood = -float('inf')
                break
            else:
                # P(w|d) = tf(w,d) / Nd
                p_w_d = tf_wd / Nd
                log_likelihood += math.log10(p_w_d)
        
        scores[doc_id] = log_likelihood
    
    # Sort by score descending
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked


def add1_laplace_smoothing(query_terms, doc_terms, doc_lengths, vocabulary):
    """
    Model 2: Add-1 (Laplace) Smoothing
    
    P(w|d) = (tf(w,d) + 1) / (Nd + |V|)
    
    where:
    - tf(w,d) = term frequency in document
    - Nd = document length
    - |V| = vocabulary size
    
    Score(q,d) = Π P(w|d) for w in q
    Log-likelihood: Σ log P(w|d) for w in q
    
    This ensures P(w|d) > 0 even if term not in document.
    """
    scores = {}
    V = len(vocabulary)
    
    for doc_id in doc_terms.keys():
        log_likelihood = 0.0
        Nd = doc_lengths[doc_id]
        
        for term in query_terms:
            tf_wd = doc_terms[doc_id].get(term, 0)
            
            # Add-1 smoothing: P(w|d) = (tf(w,d) + 1) / (Nd + |V|)
            p_w_d = (tf_wd + 1) / (Nd + V)
            log_likelihood += math.log10(p_w_d)
        
        scores[doc_id] = log_likelihood
    
    # Sort by score descending
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked


def main():
    # Open output file
    output_file = "LAB4/lab2_language_models_results.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("Lab 2: Probabilistic Unigram Language Models for Information Retrieval\n")
        f.write("="*80 + "\n\n")
        
        print("="*80)
        print("Lab 2: Probabilistic Unigram Language Models for Information Retrieval")
        print("="*80)
        print()
        
        # Load data from inverted_index.txt
        doc_terms, docs, vocabulary = load_document_terms(DOC_TERMS_PATH)
        collection_cf = compute_collection_frequencies(doc_terms, vocabulary)
        doc_lengths = compute_document_lengths(doc_terms)
        
        f.write(f"Loaded {len(docs)} documents: {docs}\n")
        f.write(f"Vocabulary size |V| = {len(vocabulary)} terms\n\n")
        f.write("Document lengths (Nd):\n")
        for doc_id in docs:
            f.write(f"  {doc_id}: Nd = {doc_lengths[doc_id]}\n")
        f.write("\n")
        
        print(f"Loaded {len(docs)} documents: {docs}")
        print(f"Vocabulary size |V| = {len(vocabulary)} terms")
        print()
        print("Document lengths (Nd):")
        for doc_id in docs:
            print(f"  {doc_id}: Nd = {doc_lengths[doc_id]}")
        print()
    
        # Test queries
        queries = {
            "q1": "large language models for information retrieval and ranking",
            "q2": "LLM for information retrieval and Ranking",
            "q3": "query Reformulation in information retrieval",
            "q4": "ranking Documents",
            "q5": "Optimizing recommendation systems with LLMs by leveraging item metadata"
        }
        
        f.write("="*80 + "\n")
        f.write("MODEL 1: Unsmoothed Language Model (MLE)\n")
        f.write("="*80 + "\n")
        f.write("Formula: P(w|d) = tf(w,d) / Nd\n")
        f.write("Score: Σ log P(w|d) for w in query\n\n")
        
        print("="*80)
        print("MODEL 1: Unsmoothed Language Model (MLE)")
        print("="*80)
        print("Formula: P(w|d) = tf(w,d) / Nd")
        print("Score: Σ log P(w|d) for w in query")
        print()
    
        for qid in sorted(queries.keys()):
            qtext = queries[qid]
            qterms = preprocess_text(qtext)
            
            f.write(f"Query {qid}: {qtext}\n")
            f.write(f"Preprocessed terms: {qterms}\n")
            
            print(f"Query {qid}: {qtext}")
            print(f"Preprocessed terms: {qterms}")
            
            ranked = mle_unsmoothed(qterms, doc_terms, doc_lengths, vocabulary)
            
            f.write(f"\nRanked documents (Log-Likelihood):\n")
            for rank, (doc_id, score) in enumerate(ranked, 1):
                if score == -float('inf'):
                    f.write(f"  {rank}. {doc_id:4s} : -∞ (query term(s) not in document)\n")
                else:
                    f.write(f"  {rank}. {doc_id:4s} : {score:10.4f}\n")
            f.write("\n" + "-"*80 + "\n\n")
            
            print(f"\nRanked documents (Log-Likelihood):")
            for rank, (doc_id, score) in enumerate(ranked, 1):
                if score == -float('inf'):
                    print(f"  {rank}. {doc_id:4s} : -∞ (query term(s) not in document)")
                else:
                    print(f"  {rank}. {doc_id:4s} : {score:10.4f}")
            print()
            print("-"*80)
            print()
    
        f.write("\n\n" + "="*80 + "\n")
        f.write("MODEL 2: Add-1 (Laplace) Smoothing\n")
        f.write("="*80 + "\n")
        f.write(f"Formula: P(w|d) = (tf(w,d) + 1) / (Nd + |V|)\n")
        f.write(f"|V| = {len(vocabulary)}\n")
        f.write("Score: Σ log P(w|d) for w in query\n\n")
        
        print("\n")
        print("="*80)
        print("MODEL 2: Add-1 (Laplace) Smoothing")
        print("="*80)
        print(f"Formula: P(w|d) = (tf(w,d) + 1) / (Nd + |V|)")
        print(f"|V| = {len(vocabulary)}")
        print("Score: Σ log P(w|d) for w in query")
        print()
        
        for qid in sorted(queries.keys()):
            qtext = queries[qid]
            qterms = preprocess_text(qtext)
            
            f.write(f"Query {qid}: {qtext}\n")
            f.write(f"Preprocessed terms: {qterms}\n")
            
            print(f"Query {qid}: {qtext}")
            print(f"Preprocessed terms: {qterms}")
            
            ranked = add1_laplace_smoothing(qterms, doc_terms, doc_lengths, vocabulary)
            
            f.write(f"\nRanked documents (Log-Likelihood):\n")
            for rank, (doc_id, score) in enumerate(ranked, 1):
                f.write(f"  {rank}. {doc_id:4s} : {score:10.4f}\n")
            f.write("\n" + "-"*80 + "\n\n")
            
            print(f"\nRanked documents (Log-Likelihood):")
            for rank, (doc_id, score) in enumerate(ranked, 1):
                print(f"  {rank}. {doc_id:4s} : {score:10.4f}")
            print()
            print("-"*80)
            print()
    
        f.write("\n\n" + "="*80 + "\n")
        f.write("MODEL 3: Good-Turing Smoothing\n")
        f.write("="*80 + "\n")
        f.write("Formula: c* = (c+1) * N_{c+1}/N_c; P(w|d) = c*/Nd; unseen: p0 = N1/(Nd*V0)\n")
        f.write("Score: Σ log P(w|d) for w in query\n\n")
        
        print("\n")
        print("="*80)
        print("MODEL 3: Good-Turing Smoothing")
        print("="*80)
        print("Formula: c* = (c+1) * N_{c+1}/N_c; P(w|d) = c*/Nd; unseen: p0 = N1/(Nd*V0)")
        print("Score: Σ log P(w|d) for w in query")
        print()
        
        for qid in sorted(queries.keys()):
            qtext = queries[qid]
            qterms = preprocess_text(qtext)
            
            f.write(f"Query {qid}: {qtext}\n")
            f.write(f"Preprocessed terms: {qterms}\n")
            
            print(f"Query {qid}: {qtext}")
            print(f"Preprocessed terms: {qterms}")
            
            ranked = good_turing_smoothing(qterms, doc_terms, doc_lengths, vocabulary)
            
            f.write(f"\nRanked documents (Log-Likelihood):\n")
            for rank, (doc_id, score) in enumerate(ranked, 1):
                if score == -float('inf'):
                    f.write(f"  {rank}. {doc_id:4s} : -∞\n")
                else:
                    f.write(f"  {rank}. {doc_id:4s} : {score:10.4f}\n")
            f.write("\n" + "-"*80 + "\n\n")
            
            print(f"\nRanked documents (Log-Likelihood):")
            for rank, (doc_id, score) in enumerate(ranked, 1):
                if score == -float('inf'):
                    print(f"  {rank}. {doc_id:4s} : -∞")
                else:
                    print(f"  {rank}. {doc_id:4s} : {score:10.4f}")
            print()
            print("-"*80)
            print()
    
        f.write("\n\n" + "="*80 + "\n")
        f.write("MODEL 4: Jelinek-Mercer Smoothing (λ=0.5)\n")
        f.write("="*80 + "\n")
        f.write(f"Formula: P(w|d) = λ*P_MLE(w|d) + (1-λ)*P_MLE(w|C)\n")
        f.write(f"λ = 0.5\n")
        f.write("Score: Σ log P(w|d) for w in query\n\n")
        
        print("\n")
        print("="*80)
        print("MODEL 4: Jelinek-Mercer Smoothing (λ=0.5)")
        print("="*80)
        print(f"Formula: P(w|d) = λ*P_MLE(w|d) + (1-λ)*P_MLE(w|C)")
        print(f"λ = 0.5")
        print("Score: Σ log P(w|d) for w in query")
        print()
        
        for qid in sorted(queries.keys()):
            qtext = queries[qid]
            qterms = preprocess_text(qtext)
            
            f.write(f"Query {qid}: {qtext}\n")
            f.write(f"Preprocessed terms: {qterms}\n")
            
            print(f"Query {qid}: {qtext}")
            print(f"Preprocessed terms: {qterms}")
            
            ranked = jelinek_mercer_smoothing(qterms, doc_terms, doc_lengths, vocabulary, collection_cf, lamb=0.5)
            
            f.write(f"\nRanked documents (Log-Likelihood):\n")
            for rank, (doc_id, score) in enumerate(ranked, 1):
                if score == -float('inf'):
                    f.write(f"  {rank}. {doc_id:4s} : -∞\n")
                else:
                    f.write(f"  {rank}. {doc_id:4s} : {score:10.4f}\n")
            f.write("\n" + "-"*80 + "\n\n")
            
            print(f"\nRanked documents (Log-Likelihood):")
            for rank, (doc_id, score) in enumerate(ranked, 1):
                if score == -float('inf'):
                    print(f"  {rank}. {doc_id:4s} : -∞")
                else:
                    print(f"  {rank}. {doc_id:4s} : {score:10.4f}")
            print()
            print("-"*80)
            print()
    
        # Compute average document length for Dirichlet smoothing
        N_avg = sum(doc_lengths.values()) / len(doc_lengths)
        mu_dirichlet = 0.3 * N_avg
        
        f.write("\n\n" + "="*80 + "\n")
        f.write("MODEL 5: Dirichlet Smoothing\n")
        f.write("="*80 + "\n")
        f.write(f"Formula: P(w|d) = (tf(w,d) + μ*P_MLE(w|C)) / (|d| + μ)\n")
        f.write(f"N_avg = {N_avg:.2f} (average document length)\n")
        f.write(f"μ = 0.3 × N_avg = 0.3 × {N_avg:.2f} = {mu_dirichlet:.2f}\n")
        f.write("Score: Σ log P(w|d) for w in query\n\n")
        
        print("\n")
        print("="*80)
        print("MODEL 5: Dirichlet Smoothing")
        print("="*80)
        print(f"Formula: P(w|d) = (tf(w,d) + μ*P_MLE(w|C)) / (|d| + μ)")
        print(f"N_avg = {N_avg:.2f} (average document length)")
        print(f"μ = 0.3 × N_avg = 0.3 × {N_avg:.2f} = {mu_dirichlet:.2f}")
        print("Score: Σ log P(w|d) for w in query")
        print()
        
        for qid in sorted(queries.keys()):
            qtext = queries[qid]
            qterms = preprocess_text(qtext)
            
            f.write(f"Query {qid}: {qtext}\n")
            f.write(f"Preprocessed terms: {qterms}\n")
            
            print(f"Query {qid}: {qtext}")
            print(f"Preprocessed terms: {qterms}")
            
            ranked, mu_used = dirichlet_smoothing(qterms, doc_terms, doc_lengths, vocabulary, collection_cf, mu=mu_dirichlet)
            
            f.write(f"\nRanked documents (Log-Likelihood):\n")
            for rank, (doc_id, score) in enumerate(ranked, 1):
                if score == -float('inf'):
                    f.write(f"  {rank}. {doc_id:4s} : -∞\n")
                else:
                    f.write(f"  {rank}. {doc_id:4s} : {score:10.4f}\n")
            f.write("\n" + "-"*80 + "\n\n")
            
            print(f"\nRanked documents (Log-Likelihood):")
            for rank, (doc_id, score) in enumerate(ranked, 1):
                if score == -float('inf'):
                    print(f"  {rank}. {doc_id:4s} : -∞")
                else:
                    print(f"  {rank}. {doc_id:4s} : {score:10.4f}")
            print()
            print("-"*80)
            print()
    
    print(f"\nRésultats enregistrés dans: {output_file}")


if __name__ == '__main__':
    main()
