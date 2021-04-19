import torch as tc
import torchtext as tt
import numpy as np
from collections import Counter

def get_tokenizer():
    tokenizer = tt.data.utils.get_tokenizer('basic_english')
    return tokenizer

def get_vocab(train_iter, tokenizer):
    counter = Counter()
    for y, X in train_iter:
        counter.update(tokenizer(X))
    vocab = tt.vocab.Vocab(counter)
    return vocab

def get_vectors():
    vectors = tt.vocab.FastText(language='en')
    return vectors

def text_pipeline(text, tokenizer, vectors, n_tokens=400):
    vectors = np.stack([vectors.get_vecs_by_tokens(token) for token in tokenizer(text)], axis=0)
    vectors = vectors[0:n_tokens]
    pad_len = n_tokens - len(vectors)
    pad_shape = (pad_len, 300)
    if pad_len > 0:
        vectors = np.concatenate([vectors, np.zeros(dtype=np.float32, shape=pad_shape)], axis=0)
    vectors = np.transpose(vectors)  # convert to NCH format.
    return vectors

def label_pipeline(y):
    d = {
        "neg": 0,
        "pos": 1
    }
    return d[y]

class ProcessedIterableDataset(tc.utils.data.IterableDataset):
    # This is cleaner than using collate_fn is the dataloader.
    # Investigation shows it is faster, as well.
    # It also lets you shuffle the data, which the dataloader itself does not support for IterableDatasets.
    def __init__(self, dataset, function):
        self.dataset = tc.utils.data.BufferedShuffleDataset(dataset, buffer_size=25000)
        self.function = function

    def __iter__(self):
        return (self.function(x,y) for x,y in self.dataset.__iter__())
