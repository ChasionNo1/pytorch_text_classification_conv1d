import numpy as np
from collections import Counter
import torchtext as tt

def get_tokenizer():
    return tt.data.utils.get_tokenizer('basic_english')

def get_vocab(train_iterator, tokenizer):
    counter = Counter()
    for y, X in train_iterator:
        counter.update(tokenizer(X))

    vectors = tt.vocab.FastText(language='en')
    vocab = tt.vocab.Vocab(counter, vectors=vectors)
    return vocab

def text_pipeline(text, tokenizer, vocab, n_tokens=400):
    vectors = np.stack([vocab.vectors[vocab.stoi[token]] for token in tokenizer(text)], axis=0)
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
