import numpy as np
import nltk
import pandas as pd
from nltk.stem import PorterStemmer

Porter = PorterStemmer()
stopwords = nltk.corpus.stopwords.words('english')
inverted_file = "LAB4/results/inverted_index_weighted.txt"

data = { "D1":[], "D2":[], "D3":[], "D4":[], "D5":[], "D6":[] }

with open(inverted_file, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 4:
            term, doc, tf, tfidf = parts
            data[doc].append((term, int(tf)))


vocab_doc = {}
collection_counts = {}   # cf(w,C)

for doc, terms in data.items():
    vocab_doc[doc] = {}
    for (t, tf) in terms:
        vocab_doc[doc][t] = vocab_doc[doc].get(t, 0) + tf
        collection_counts[t] = collection_counts.get(t, 0) + tf

V_collection = set(collection_counts.keys())
collection_length = sum(collection_counts.values())


query = {
    "q1": "large language models for information retrieval and ranking",
    "q2": "LLM for information retrieval and Ranking", 
    "q3":"query Reformulation in information retrieval",
    "q4": "ranking Documents", 
    "q5": "Optimizing recommendation systems with LLMs by leveraging item metadata"
}

def preprocess_query(q):
    q = q.lower()
    tokens = q.split()
    tokens = [Porter.stem(t) for t in tokens if t not in stopwords]
    return tokens

query_tokens = {qid: preprocess_query(q) for qid, q in query.items()}

def P_MLE(w, d):
    tf = vocab_doc[d].get(w, 0)
    return tf / sum(vocab_doc[d].values()) if sum(vocab_doc[d].values()) > 0 else 0

def P_Laplace(w, d):
    tf = vocab_doc[d].get(w, 0)
    nd = sum(vocab_doc[d].values())
    V = len(V_collection)
    
    return (tf + 1) / (nd + V)

def P_JM(w, d, lam=0.4):
    tf = vocab_doc[d].get(w, 0)
    nd = sum(vocab_doc[d].values())
    PwC = collection_counts.get(w, 0) / collection_length
    PwD = tf / nd if nd > 0 else 0
    return lam * PwD + (1 - lam) * PwC

def calcul_mu():
    N_avg = 0
    for d in ["D1","D2","D3","D4","D5","D6"]:
        N_avg = N_avg + sum(vocab_doc[d].values())
    return 0.3*(N_avg/6)

def P_Dirichlet(w, d, mu):
    tf = vocab_doc[d].get(w, 0)
    nd = sum(vocab_doc[d].values())
    PwC = collection_counts.get(w, 0) / collection_length
    return (tf + mu * PwC) / (nd + mu)

# Version normale de Good-Turing (formule exacte)
def P_GoodTuring_normal(w, d):
    """
    Version normale de Good-Turing selon la formule:
    freq*(s) = (freq(s) + 1) * n_{s+1} / n_s
    """
    # Fréquence du mot dans le document
    freq_s = vocab_doc[d].get(w, 0)
    
    # Calculer n_r pour ce document
    doc_counts = vocab_doc[d]
    n_r = {}
    for tf in doc_counts.values():
        n_r[tf] = n_r.get(tf, 0) + 1
    
    # Si mot non vu
    if freq_s == 0:
        n1 = n_r.get(1, 0)
        N_d = sum(vocab_doc[d].values())
        if N_d > 0:
            # Probabilité pour les mots non vus = n1/N
            return n1 / N_d
        return 0
    
    # Si mot vu
    n_s = n_r.get(freq_s, 0)
    n_s_plus_1 = n_r.get(freq_s + 1, 0)
    
    # Appliquer formule Good-Turing
    if n_s > 0:
        freq_star = (freq_s + 1) * n_s_plus_1 / n_s
    else:
        freq_star = freq_s
    
    # Convertir en probabilité
    N_d = sum(vocab_doc[d].values())
    if N_d > 0:
        return freq_star / N_d
    return 0

# Version approximation de Good-Turing (comme dans l'image)
def P_GoodTuring_approx(w, d):
    """
    Version approximation de Good-Turing selon les 4 étapes:
    1. Masse pour mots jamais vus: p0 = n1/N
    2. Masse restante: 1 - p0
    3. Calcul probabilités avec ajustement
    4. Produit pour séquence
    """
    # Fréquence du mot dans le document
    freq_s = vocab_doc[d].get(w, 0)
    
    # Calculer n_r pour ce document
    doc_counts = vocab_doc[d]
    n_r = {}
    for tf in doc_counts.values():
        n_r[tf] = n_r.get(tf, 0) + 1
    
    # Étape 1: Calculer p0 = n1/N
    n1 = n_r.get(1, 0)
    N_d = sum(vocab_doc[d].values())
    
    if N_d == 0:
        return 0
    
    p0 = n1 / N_d
    
    # Étape 2: Masse restante pour les mots vus
    mass_remaining = 1 - p0
    
    # Étape 3: Calculer les probabilités
    if freq_s == 0:
        # Mot non vu - probabilité uniforme parmi tous mots possibles
        # Approximation: utiliser le vocabulaire de la collection
        V_total = len(V_collection)
        V_seen = len(doc_counts)
        V_unseen = V_total - V_seen
        
        if V_unseen > 0:
            return p0 / V_unseen
        else:
            return p0
    
    else:
        # Mot vu: P*(w) = (1 - p0) * (freq(w) / N)
        return mass_remaining * (freq_s / N_d)

# Fonction pour décider quelle version utiliser
def P_GoodTuring(w, d, use_approx=True):
    """
    Fonction qui choisit automatiquement entre version normale et approximation.
    Decision rule: use approximation si n0 est inconnu ou données insuffisantes
    """
    # Calculer les statistiques du document
    doc_counts = vocab_doc[d]
    n_r = {}
    for tf in doc_counts.values():
        n_r[tf] = n_r.get(tf, 0) + 1
    
    N_d = sum(vocab_doc[d].values())
    
    # Vérifier si on peut utiliser la version normale
    freq_s = vocab_doc[d].get(w, 0)
    
    if freq_s > 0:
        n_s = n_r.get(freq_s, 0)
        n_s_plus_1 = n_r.get(freq_s + 1, 0)
        
        # Version normale requiert n_s > 0 et souvent n_{s+1} > 0
        # Si données insuffisantes, utiliser approximation
        if use_approx or n_s == 0 or (freq_s > 0 and n_s_plus_1 == 0):
            # Quand n_{s+1} = 0, formule normale donne probabilité 0
            # Dans ce cas, mieux vaut utiliser l'approximation
            return P_GoodTuring_approx(w, d)
        else:
            return P_GoodTuring_normal(w, d)
    else:
        # Pour les mots non vus, l'approximation est plus stable
        return P_GoodTuring_approx(w, d)

def score_document(query_tokens, d, model="mle", use_approx=True):
    score = 1
    for w in query_tokens:
        if model == "mle":
            p = P_MLE(w, d)
        elif model == "laplace":
            p = P_Laplace(w, d)
        elif model == "jm":
            p = P_JM(w, d, lam=0.4)
        elif model == "dir":
            p = P_Dirichlet(w, d, mu=calcul_mu())
        elif model == "gt":
            p = P_GoodTuring(w, d, use_approx)
        elif model == "gt_normal":
            p = P_GoodTuring_normal(w, d)
        elif model == "gt_approx":
            p = P_GoodTuring_approx(w, d)
        else:
            raise ValueError("Unknown model")
        
        # Éviter le zéro pour le calcul du produit
        if p == 0:
            p = 1e-10
        
        score *= p

    return score


def rank_documents(q_tokens, model, use_approx=True):
    results = {}
    for d in data.keys():
        results[d] = score_document(q_tokens, d, model=model, use_approx=use_approx)

    ranked = sorted(results.items(), key=lambda x: x[1], reverse=True)
    return ranked

# Tester les deux versions
models = ["mle", "laplace", "jm", "dir", "gt", "gt_normal", "gt_approx"]

for qid, tokens in query_tokens.items():
    print("\n" + "="*40)
    print(f"Query: {qid} → {tokens}")
    print("="*40)

    for m in models:
        if m == "gt":
            # Version avec décision automatique
            print(f"\n--- Model: {m.upper()} (Auto-decision) ---")
            ranking = rank_documents(tokens, m, use_approx=True)
        elif m == "gt_normal":
            print(f"\n--- Model: GOOD-TURING NORMAL ---")
            ranking = rank_documents(tokens, m, use_approx=False)
        elif m == "gt_approx":
            print(f"\n--- Model: GOOD-TURING APPROX ---")
            ranking = rank_documents(tokens, m, use_approx=True)
        else:
            print(f"\n--- Model: {m.upper()} ---")
            ranking = rank_documents(tokens, m, use_approx=True)
        
        for doc, score in ranking:
            print(f"{doc}  →  score = {score:.10f}")

# Comparaison détaillée pour une requête
print("\n" + "="*60)
print("COMPARAISON DÉTAILLÉE GOOD-TURING NORMAL vs APPROX")
print("="*60)

# Exemple avec une requête
qid = "q1"
tokens = query_tokens[qid]
print(f"\nRequête: {qid} = {tokens}")

for d in data.keys():
    print(f"\nDocument {d}:")
    print("-" * 30)
    
    # Calculer pour chaque mot
    for w in tokens:
        p_normal = P_GoodTuring_normal(w, d)
        p_approx = P_GoodTuring_approx(w, d)
        freq = vocab_doc[d].get(w, 0)
        
        # Décision: quelle version utiliser?
        doc_counts = vocab_doc[d]
        n_r = {}
        for tf in doc_counts.values():
            n_r[tf] = n_r.get(tf, 0) + 1
        
        if freq > 0:
            n_s = n_r.get(freq, 0)
            n_s_plus_1 = n_r.get(freq + 1, 0)
            
            decision = "APPROX" if (n_s == 0 or n_s_plus_1 == 0) else "NORMAL"
        else:
            decision = "APPROX"
        
        print(f"  {w} (freq={freq}): normal={p_normal:.6f}, approx={p_approx:.6f} → décision: {decision}")