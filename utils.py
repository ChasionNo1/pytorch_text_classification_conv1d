import torch as tc
import torchtext as tt
import numpy as np
from collections import Counter
from functools import partial


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


def collate_batch(batch, map_fn):
    X, Y = zip(*[map_fn(y, x) for y,x in batch])
    X = tc.Tensor(list(X)).float()
    Y = tc.Tensor(list(Y)).long()
    return X, Y


def get_dataloaders(map_fn, batch_size):
    training_data = tt.datasets.IMDB(root='data', split='train')
    test_data = tt.datasets.IMDB(root='data', split='test')

    training_data = tc.utils.data.BufferedShuffleDataset(training_data, buffer_size=25000)
    test_data = tc.utils.data.BufferedShuffleDataset(test_data, buffer_size=25000)

    collate_fn = partial(collate_batch, map_fn=map_fn)

    train_dataloader = tc.utils.data.DataLoader(training_data, batch_size=batch_size, collate_fn=collate_fn)
    test_dataloader = tc.utils.data.DataLoader(test_data, batch_size=batch_size, collate_fn=collate_fn)

    return train_dataloader, test_dataloader
