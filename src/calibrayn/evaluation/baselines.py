import math
from collections import Counter, defaultdict
from statistics import fmean

from calibrayn.text import content_words


class Bm25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        tokens = [content_words(text) for text in corpus]
        self.size = len(tokens)
        self.lengths = [len(words) for words in tokens]
        self.average = fmean(self.lengths) if self.lengths and any(self.lengths) else 1.0
        self.postings = defaultdict(list)
        for index, words in enumerate(tokens):
            for term, count in Counter(words).items():
                self.postings[term].append((index, count))
        self.idf = {
            term: math.log(1.0 + (self.size - len(found) + 0.5) / (len(found) + 0.5))
            for term, found in self.postings.items()
        }

    def scores(self, query):
        totals = defaultdict(float)
        for term in set(content_words(query)):
            weight = self.idf.get(term, 0.0)
            for index, count in self.postings.get(term, ()):
                length = self.lengths[index]
                denominator = count + self.k1 * (1.0 - self.b + self.b * length / self.average)
                totals[index] += weight * count * (self.k1 + 1.0) / denominator
        return totals

    def ranks_of(self, query, gold):
        totals = self.scores(query)
        ranks = []
        for index in gold:
            if index not in totals:
                ranks.append(None)
                continue
            ahead = sum(1 for value in totals.values() if value > totals[index])
            ranks.append(ahead + 1)
        return tuple(ranks)
