import re

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

    @staticmethod
    def get_stats_for_words(ids):
        counts = {}
        for word_id in ids:
            for pair in zip(word_id, word_id[1:]):
                counts[pair] = counts.get(pair, 0) + 1
        return counts

    def train_tokenizer(self, text: str):
        if self.pattern == "":
            tokens = text.encode('utf-8')
            tokens = list(map(int, tokens))
            ids = tokens[:]

            vocab_size = self.vocab_size
            num_merges = vocab_size - 256
            merges = {}

            for i in range(num_merges):
                stats = self.get_stats(ids)
                if not stats:
                    print("Probably, vocab_size is too small")
                    break

                pair = max(stats, key=stats.get)
                idx = 256 + i
                # print(f"merging {pair} into a token {idx}")
                ids = self.merge(ids, pair, idx)
                merges[pair] = idx

            vocab = {idx: bytes([idx]) for idx in range(256)}
            for (p0, p1), idx in merges.items():
                vocab[idx] = vocab[p0] + vocab[p1]

            return vocab, merges

        words = re.findall(self.pattern, text)
        token_words = []
        for word in words:
            token_words.append(word.encode('utf-8'))
            token_words[-1] = list(map(int, token_words[-1]))
        ids = token_words[:]

        vocab_size = self.vocab_size
        num_merges = vocab_size - 256
        merges = {}

        for i in range(num_merges):
            stats = self.get_stats_for_words(ids)
            if not stats:
                print("Probably, vocab_size is too small")
                break

            pair = max(stats, key=stats.get)
            idx = 256 + i
            # print(f"merging {pair} into a token {idx}")
            for word_ids in ids:
                word_ids = self.merge(word_ids, pair, idx)
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


tokenizer = Tokenizer(1024, r'\w+|[.,!?;]')
text = "Among the passengers travelling home by train from Florence there was a certain Miss Bradley.I only noticed her when passing down the corridor, because of her really remarkable plainness. She was rather a large, awkward woman of about thirty-five with a big, red nose, and large spectacles.Later on, when I went to the dining-car, Miss Bradley was already seated, and the attendant placed me opposite her.I think we may have exchanged half a dozen words at dinner, when passing one another the sugar or the bread. But they were certainly all we exchanged, and after we left the dining-car, I did not see Miss Bradley again until we reached Calais Maritime.And then our acquaintance really began, and it began entirely on my initiative. There were plenty of porters, and I called one without difficulty from the window of the train. But as I got off, I saw Miss Bradley standing on the platform with two large very old suit-cases. The porters were passing her by.I am quite sure that had she been an even slightly attractive woman, I should not have gone up to her, but she was so ugly, and looked so helpless that I approached her, and said: My porter has a barrow. Would you like him to put your cases on it too? Miss Bradley turned and looked at me.Oh — thank you. It is very kind of you.My porter, without great enthusiasm, added her luggage to mine; and in a few minutes we found ourselves on board the Channel ferry.Before the boat had been under way for ten minutes, I realized that Miss Bradley was a remarkable bore. Shyly and hesitantly she kept on talking about nothing, and made no remark worth taking notice of.I learned that she had been in Italy a fortnight, visiting her sister who was married to an Italian. She had never been out of England before.I did not look forward to travelling to London with her for another four hours, so excusing myself I went along to the booking-office on board the boat and booked myself a seat on the Golden Arrow.Miss Bradley was travelling by the ordinary boat train, so this would mean that we should part at Dover.At Dover I hired one of the crew to carry our luggage.Normally, passengers for the Golden Arrow are dealt with by the customs first, as the train leaves twenty minutes before the ordinary boat train. When the boy asked if we were going on the Golden Arrow, I hesitated and then said Yes.It was too difficult to explain that one of us was and one of us wasn't, and then it would get Miss Bradley through the customs quickly.As we went towards the Customs Hall, I explained carefully to her that my train left before hers, but that I would see her through the customs; the boy would then take the luggage to our trains, and she could sit comfortably in hers till it left. Miss Bradley said, Oh, thank you very much.The boy, of course, had put our suit-cases together on the counter, and Miss Bradley and I went and stood before them. In due course the customs examiner reached us, looked at the four suitcases in that human X-ray manner which customs examiners must practise night and morning, and said, This is all yours?I was not quite sure whether he was speaking to me, or me and Miss Bradley. So I replied, Well — mine and this lady's.The examiner said, But you're together?For the moment, I said rather foolishly, smiling at Miss Bradley.Yes, said the customs man patiently. But are you travelling together? Is this your joint luggage?Well, no. Not exactly. We're just sharing a porter I pointed my cases out. I had nothing to declare, and declared it. Without asking me to open them, the examiner chalked the cases and then, instead of moving to my left and dealing with Miss Bradley, moved to the right, and began X-raying somebody else's luggage.The boy took my cases off the counter. I hesitated for a moment, but then decided it was no use waiting for Miss Bradley since we were about to part, so I said:Well, I'll say good-bye now, and go and find my train. I expect the examiner'll come back and do you next. The porter will stay and bring our luggage up to the trains when you're through. Good-bye.Miss Bradley said, Oh... good-bye and thank you so much. We shook hands and I left.I found my seat in the Golden Arrow and began to read.It must have been about twenty minutes later that I suddenly realized the train was due to leave in five minutes and that the porter had not yet brought my luggage. I was just going to look for him when he appeared, breathing heavily, with my suit-cases. I asked him rather what he had been doing.The lady is still there, said the boy, and will be for some time, I think. They are going through her things properly.But why?Well, they'd found forty watches when I came away, and that was only the start, so I thought maybe you wouldn't want me to wait.I have often wondered whether, when Miss Bradley stood so helplessly on the platform at Calais, she had already chosen me as the person to come to her rescue, or whether she was just sure that somebody would.Looking back, I think, she must have chosen me. I am fairly sure of that though exactly how, I have never been clear. I am quite sure she never made the slightest effort to make my acquaintance."
vocab, merges = tokenizer.train_tokenizer(text)
smth = tokenizer.encode("i love you and hate him, because 2007 years ago he`s wife killed me!", merges)
print(smth)
print(tokenizer.decode(smth, vocab))
