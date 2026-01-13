class Tokenizer:
    def __init__(self, vocab_size: int, pattern: str):
        self.vocab_size = vocab_size
        self.pattern = pattern

    @staticmethod
    def get_stats(ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    @staticmethod
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

    def train_tokenizer(self, text: str):
        tokens = text.encode('utf-8')
        tokens = list(map(int, tokens))
        ids = tokens[:]

        vocab_size = self.vocab_size
        num_merges = vocab_size - 256
        merges = {}

        for i in range(num_merges):
            stats = self.get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            # print(f"merging {pair} into a token {idx}")
            ids = self.merge(ids, pair, idx)
            merges[pair] = idx

        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]

        return vocab, merges

    def decode(self, ids, vocab):
        tokens = b"".join(vocab[idx] for idx in ids)
        text = tokens.decode('utf-8', errors='replace')
        return text

    def encode(self, text, merges):
        tokens = text.encode('utf-8')
        while len(tokens) >= 2:
            stats = self.get_stats(tokens)
            pair = min(stats, key=lambda p: merges.get(p, float('inf')))
            if pair not in merges:
                break
            idx = merges[pair]
            tokens = self.merge(tokens, pair, idx)
        return tokens
