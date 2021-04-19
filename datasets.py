import torch as tc
import torchtext as tt
from collections import Counter
import numpy as np

# it may be better to switch to using collate_fn as in tutorial, but this looks cleaner.
# keeping it for now.

class IMDBDatasetEmbedded(tc.utils.data.IterableDataset):
    def __init__(self, root, train):
        self.root = root
        self.tokenizer = tt.data.utils.get_tokenizer('basic_english')
        self.vocab = self.get_vocab(self.root, self.tokenizer)
        self.dataset = tc.utils.data.BufferedShuffleDataset(
            tt.datasets.IMDB(split='train' if train else 'test'),
            buffer_size=50000
        )
        self.n_tokens = 400

    @staticmethod
    def get_vocab(root, tokenizer):
        train_iter = tt.datasets.IMDB(root=root, split='train')

        counter = Counter()
        for y, X in train_iter:
            counter.update(tokenizer(X))

        vectors = tt.vocab.FastText(language='en')
        vocab = tt.vocab.Vocab(counter, vectors=vectors)
        return vocab

    def text_pipeline(self, text, tokenizer, vocab):
        vectors = np.stack([vocab.vectors[vocab.stoi[token]] for token in tokenizer(text)], axis=0)
        vectors = vectors[0:self.n_tokens]
        pad_len = self.n_tokens-len(vectors)
        pad_shape = (pad_len, 300)
        if pad_len > 0:
            vectors = np.concatenate([vectors, np.zeros(dtype=np.float32, shape=pad_shape)], axis=0)
        vectors = np.transpose(vectors) # convert to NCH format.
        return vectors

    def label_pipeline(self, y):
        d = {
            "neg": 0,
            "pos": 1
        }
        return d[y]

    def __iter__(self):
        fn = lambda X, y: (
            self.text_pipeline(X, self.tokenizer, self.vocab),
            self.label_pipeline(y)
        )
        return (fn(X,y) for y, X in self.dataset.__iter__())
