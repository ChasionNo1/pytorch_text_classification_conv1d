import torch as tc
import torchtext as tt
from functools import partial
from datasets import IMDBDatasetEmbedded
#from utils import get_tokenizer, get_vocab, text_pipeline, label_pipeline
from classifier import Conv1dTextClassifier
from runner import Runner

"""
# Download datasets. These text datasets are IterableDatasets.
training_data = tc.utils.data.BufferedShuffleDataset(tt.datasets.IMDB(split='train'), buffer_size=25000)
test_data = tc.utils.data.BufferedShuffleDataset(tt.datasets.IMDB(split='test'), buffer_size=25000)
"""

training_data = IMDBDatasetEmbedded(root='data', train=True)
test_data = IMDBDatasetEmbedded(root='data', train=False)

"""
# Create a map to be applied to each batch item.
tokenizer = get_tokenizer()
vocab = get_vocab(training_data, tokenizer)
text_preprocessing = partial(text_pipeline, tokenizer=tokenizer, vocab=vocab)
def collate_batch(batch):
    x_list = []
    y_list = []
    for y, x in batch:
        x_list.append(text_preprocessing(x))
        y_list.append(label_pipeline(y))
    X = tc.tensor(x_list, dtype=tc.float32)
    Y = tc.tensor(y_list, dtype=tc.int64)
    return X, Y
"""

tokenizer = training_data.tokenizer
vocab = training_data.vocab
text_preprocessing = partial(training_data.text_pipeline, tokenizer=tokenizer, vocab=vocab)

"""
# Create data loaders to batch, shuffle, and map the data.
batch_size = 50
train_dataloader = tc.utils.data.DataLoader(training_data, batch_size=batch_size, collate_fn=collate_batch)
test_dataloader = tc.utils.data.DataLoader(test_data, batch_size=batch_size, collate_fn=collate_batch)
"""

batch_size = 50
train_dataloader = tc.utils.data.DataLoader(training_data, batch_size=batch_size)
test_dataloader = tc.utils.data.DataLoader(test_data, batch_size=batch_size)

# Device.
device = "cuda" if tc.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Model from Yoon Kim, 2014 - 'Convolutional Neural Networks for Text Classification'.
model = Conv1dTextClassifier(num_classes=2).to(device)
print(model)

criterion = tc.nn.CrossEntropyLoss()
optimizer = tc.optim.Adadelta(model.parameters(), lr=1.0)

try:
    model.load_state_dict(tc.load("model.pth"))
    optimizer.load_state_dict(tc.load("optimizer.pth"))
    print('successfully reloaded checkpoint. continuing training...')
except Exception:
    print('no checkpoint found. training from scratch...')

# Instantiate a runner to run the training epochs and monitor performance.
runner = Runner(max_epochs=1, verbose=True)
runner.run(model, train_dataloader, test_dataloader, device, criterion, optimizer)
print("All done!")

# Run a demo
model.eval()
hypothetical_review = "The movie was good, but there were some over the top moments especially in the action scenes. Overall B+."
x = tc.from_numpy(text_preprocessing(hypothetical_review))
y_logits = model.forward(tc.unsqueeze(x, dim=0))
y_probs = tc.nn.Softmax(dim=-1)(y_logits)
print(hypothetical_review)
print(y_probs)