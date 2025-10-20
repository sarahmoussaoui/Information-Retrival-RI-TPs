# Define your words and order explicitly
words_order = [
    ('D1', 'ndcg@10'),
    ('D2', 'gpt-4'),
    ('D3', 'e.g.'),
    ('D3', 'model'),
    ('D4', 'llm'),
    ('D5', 'graph'),
    ('D5', 'model'),
    ('D6', 'descript'),
    ('D6', 'gpt-3.5'),
    ('D1', 'languag'),
    ('D2', 'languag'),
    ('D3', 'languag'),
    ('D4', 'languag'),
    ('D5', 'languag'),
    ('D6', 'languag'),
    ('D2', 'chatgpt')
]

matches = []

# Read and collect lines that match the pairs
with open("LAB2/results/descriptor_weighted.txt", "r") as f_in:
    lines = f_in.readlines()
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue
        doc, term = parts[0].strip(), parts[1].strip()
        for w_doc, w_term in words_order:
            if doc == w_doc and term == w_term:
                matches.append((w_doc, w_term, line))
                break

# Write results in the same custom order
with open("LAB2/results/research_results.txt", "w") as f_out:
    for w_doc, w_term in words_order:
        for doc, term, line in matches:
            if doc == w_doc and term == w_term:
                f_out.write(line)
                break
