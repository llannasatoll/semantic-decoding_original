import numpy as np
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")

INIT = {
    "original": ["i", "we", "she", "he", "they", "it"],
    "llama3": ["I", "We", "She", "He", "They", "It"],
    "llama70b": ["I", "We", "She", "He", "They", "It"],
    "opt": ["I", "We", "She", "He", "They", "It"],
    "gpt": ["i</w>", "we</w>", "she</w>", "he</w>", "they</w>", "it</w>"],
}

PROMPT = {
    "llama3": ['<|begin_of_text|>', 'I', "'ll", 'Ġbriefly', 'Ġdescribe', 'Ġthe', 'Ġscene', 'Ġfrom', 'Ġthe', 'Ġmovie', 'ĠI', "'m", 'Ġcurrently', 'Ġwatching', '.'],
    "llama70b": ['<|begin_of_text|>', 'I', "'ll", 'Ġbriefly', 'Ġdescribe', 'Ġthe', 'Ġscene', 'Ġfrom', 'Ġthe', 'Ġmovie', 'ĠI', "'m", 'Ġcurrently', 'Ġwatching', '.'],
    "opt": ['</s>', 'I', "'ll", 'Ġbriefly', 'Ġdescribe', 'Ġthe', 'Ġscene', 'Ġfrom', 'Ġthe', 'Ġmovie', 'ĠI', "'m", 'Ġcurrently', 'Ġwatching', '.'],
    "original": ["i'll</w>", "briefly</w>", "describe</w>", "the</w>", "scene</w>", "from</w>", "the</w>", "movie</w>", "i'm</w>", "currently</w>", "watching</w>", "."]
}

STOPWORDS = {
    "is",
    "does",
    "s",
    "having",
    "doing",
    "these",
    "shan",
    "yourself",
    "other",
    "are",
    "hasn",
    "at",
    "for",
    "while",
    "down",
    "hadn't",
    "until",
    "above",
    "during",
    "each",
    "now",
    "have",
    "won't",
    "once",
    "why",
    "here",
    "ourselves",
    "to",
    "over",
    "into",
    "who",
    "that",
    "myself",
    "he",
    "themselves",
    "were",
    "against",
    "about",
    "some",
    "has",
    "but",
    "ma",
    "their",
    "this",
    "there",
    "with",
    "that'll",
    "shan't",
    "wouldn't",
    "a",
    "those",
    "you'll",
    "ll",
    "few",
    "couldn",
    "an",
    "d",
    "weren't",
    "doesn",
    "own",
    "won",
    "didn",
    "what",
    "when",
    "in",
    "below",
    "where",
    "it's",
    "most",
    "just",
    "you're",
    "yourselves",
    "too",
    "don't",
    "she's",
    "didn't",
    "hasn't",
    "isn",
    "mustn't",
    "of",
    "did",
    "how",
    "himself",
    "aren",
    "if",
    "very",
    "or",
    "weren",
    "it",
    "be",
    "itself",
    "doesn't",
    "my",
    "o",
    "no",
    "isn't",
    "before",
    "after",
    "off",
    "was",
    "can",
    "the",
    "been",
    "her",
    "him",
    "wasn't",
    "ve",
    "through",
    "needn't",
    "because",
    "nor",
    "will",
    "m",
    "t",
    "out",
    "on",
    "she",
    "all",
    "then",
    "than",
    "mightn't",
    "hers",
    "herself",
    "only",
    "should",
    "re",
    "ain",
    "wasn",
    "aren't",
    "couldn't",
    "they",
    "hadn",
    "had",
    "more",
    "and",
    "under",
    "shouldn't",
    "any",
    "y",
    "don",
    "from",
    "so",
    "whom",
    "as",
    "mustn",
    "between",
    "up",
    "do",
    "both",
    "such",
    "our",
    "its",
    "which",
    "not",
    "haven't",
    "needn",
    "by",
    "should've",
    "again",
    "shouldn",
    "his",
    "me",
    "further",
    "yours",
    "am",
    "your",
    "haven",
    "wouldn",
    "being",
    "ours",
    "you",
    "i",
    "theirs",
    "mightn",
    "same",
    "we",
    "you've",
    "them",
    "you'd",
}


def get_nucleus(probs, nuc_mass, nuc_ratio):
    """identify words that constitute a given fraction of the probability mass"""
    nuc_ids = []
    while len(nuc_ids) == 0:
        nuc_ids = np.where(probs >= np.max(probs) * nuc_ratio)[0]
        nuc_pairs = sorted(zip(nuc_ids, probs[nuc_ids]), key=lambda x: -x[1])
        sum_mass = np.cumsum([x[1] for x in nuc_pairs])
        cutoffs = np.where(sum_mass >= nuc_mass)[0]
        if len(cutoffs) > 0:
            nuc_pairs = nuc_pairs[: cutoffs[0] + 1]
        nuc_ids = [x[0] for x in nuc_pairs]
        nuc_mass += 0.01
        nuc_ratio -= 0.01
    return nuc_ids


def in_context(word, context):
    """test whether [word] or a stem of [word] is in [context]"""
    stem_context = [stemmer.stem(x) for x in context]
    stem_word = stemmer.stem(word)
    return stem_word in stem_context or stem_word in context


def context_filter(proposals, context):
    """filter out words that occur in a context to prevent repetitions"""
    cut_words = []
    cut_words.extend(
        [context[i + 1] for i, word in enumerate(context[:-1]) if word == context[-1]]
    )  # bigrams
    cut_words.extend(
        [x for x in proposals if x not in STOPWORDS and in_context(x, context)]
    )  # unigrams
    return [x for x in proposals if x not in cut_words]


class LanguageModel:
    """class for generating word sequences using a language model"""

    def __init__(self, model, vocab, prompt, nuc_mass=1.0, nuc_ratio=0.0):
        self.model = model
        self.has_prompt = prompt
        vocab = set(vocab)
        self.ids = {i for word, i in self.model.word2id.items() if word in vocab}
        self.nuc_mass, self.nuc_ratio = nuc_mass, nuc_ratio

    def ps(self, contexts):
        """get probability distributions over the next words for each context"""
        context_arr = self.model.get_context_array(contexts)
        probs = self.model.get_probs(context_arr)
        return probs[:, len(contexts[0]) - 1]

    def beam_propose(self, beam, context_words):
        """get possible extension words for each hypothesis in the decoder beam"""
        if len(beam) == 1:
            # nuc_words = [
            #     w for w in INIT[self.model.llm] if self.model.word2id[w] in self.ids
            # ]
            nuc_words = ["" for _ in range()]
            nuc_logprobs = np.log(np.ones(len(nuc_words)) / len(nuc_words))
            return [(nuc_words, nuc_logprobs)]
        else:
            if (len(beam[0].words) < 100) and self.has_prompt:
                contexts = [PROMPT[self.model.llm] + hyp.words for hyp in beam]
            else:
                contexts = [hyp.words[-100:] for hyp in beam]

            beam_probs = self.ps(contexts)
            beam_nucs = []
            for context, probs in zip(contexts, beam_probs):
                nuc_ids = get_nucleus(
                    probs, nuc_mass=self.nuc_mass, nuc_ratio=self.nuc_ratio
                )
                nuc_words = [self.model.vocab[i] for i in nuc_ids if i in self.ids]
                nuc_words_filtered = context_filter(nuc_words, context[-20:])
                if len(nuc_words_filtered) == 0:
                    nuc_words_filtered = nuc_words
                nuc_logprobs = np.log(
                    [probs[self.model.word2id[w]] for w in nuc_words_filtered]
                )
                beam_nucs.append((nuc_words, nuc_logprobs))
            return beam_nucs
