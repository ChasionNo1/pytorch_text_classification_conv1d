import torch as tc
import torchtext as tt
from collections import Counter
import numpy as np


class IMDBDatasetEmbedded(tc.utils.data.IterableDataset):
    # This is cleaner than using collate_fn is the dataloader.
    # Investigation shows it is faster, as well.
    def __init__(self, root, train):
        self.root = root
        self.tokenizer = tt.data.utils.get_tokenizer('basic_english')
        self.vocab = self.get_vocab(self.root, self.tokenizer)
        self.vectors = self.get_vectors()
        self.dataset = tc.utils.data.BufferedShuffleDataset(
            tt.datasets.IMDB(split='train' if train else 'test'),
            buffer_size=50000
        )

    @staticmethod
    def get_vocab(root, tokenizer):
        train_iter = tt.datasets.IMDB(root=root, split='train')

        counter = Counter()
        for y, X in train_iter:
            counter.update(tokenizer(X))

        vocab = tt.vocab.Vocab(counter)
        return vocab

    @staticmethod
    def get_vectors():
        vectors = tt.vocab.FastText(language='en')
        return vectors

    @staticmethod
    def text_pipeline(self, text, tokenizer, vocab, vectors, n_tokens=400):
        vectors = np.stack([vectors[vocab.stoi[token]] for token in tokenizer(text)], axis=0)
        vectors = vectors[0:n_tokens]
        pad_len = n_tokens-len(vectors)
        pad_shape = (pad_len, 300)
        if pad_len > 0:
            vectors = np.concatenate([vectors, np.zeros(dtype=np.float32, shape=pad_shape)], axis=0)
        vectors = np.transpose(vectors) # convert to NCH format.
        return vectors

    @staticmethod
    def label_pipeline(y):
        d = {
            "neg": 0,
            "pos": 1
        }
        return d[y]

    def __iter__(self):
        fn = lambda X, y: (
            self.text_pipeline(X, self.tokenizer, self.vocab, self.vectors),
            self.label_pipeline(y)
        )
        return (fn(X,y) for y, X in self.dataset.__iter__())