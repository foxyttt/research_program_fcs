def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and pair[0] == ids[i] and pair[1] == ids[i + 1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

def train_tokenizer(text):
    tokens = text.encode('utf-8')
    tokens = list(map(int, tokens))
    ids = tokens[:]

    vocab_size = 512
    num_merges = vocab_size - 256
    merges = {}

    for i in range(num_merges):
        stats = get_stats(ids)
        pair = max(stats, key=stats.get)
        idx = 256 + i
        #print(f"merging {pair} into a token {idx}")
        ids = merge(ids, pair, idx)
        merges[pair] = idx

    vocab = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]

    return vocab, merges

def decode(ids, vocab):
    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode('utf-8', errors='replace')
    return text

def encode(text, merges):
    tokens = text.encode('utf-8')
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, key= lambda p: merges.get(p, float('inf')))
        if pair not in merges:
            break
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens




