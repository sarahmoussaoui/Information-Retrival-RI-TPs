# Define your words and order explicitly
words_order = [
    ('D1.txt', 'ndcg@10'),
    ('D2.txt', 'gpt-4'),
    ('D3.txt', 'e.g.'),
    ('D3.txt', 'model'),
    ('D4.txt', 'llm'),
    ('D5.txt', 'graph'),
    ('D5.txt', 'model'),
    ('D6.txt', 'descript'),
    ('D6.txt', 'gpt-3.5'),
    ('D1.txt', 'languag'),
    ('D2.txt', 'languag'),
    ('D3.txt', 'languag'),
    ('D4.txt', 'languag'),
    ('D5.txt', 'languag'),
    ('D6.txt', 'languag'),
    ('D2.txt', 'chatgpt')
]

matches = []

# Read and collect lines that match the pairs
with open("./LAB2/results/descriptor_weighted.txt", "r") as f_in:
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
with open("./LAB2/results/research_results.txt", "w") as f_out:
    for w_doc, w_term in words_order:
        for doc, term, line in matches:
            if doc == w_doc and term == w_term:
                f_out.write(line)
                break
