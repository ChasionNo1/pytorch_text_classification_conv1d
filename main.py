import torch as tc
from datasets import IMDBDatasetEmbedded
from classifier import Conv1dTextClassifier
from runner import Runner
import numpy as np

# IMDB with preprocessing by embedding words using FastText.
training_data = IMDBDatasetEmbedded(root="data", train=True)
test_data = IMDBDatasetEmbedded(root="data", train=False)

# Create data loaders.
batch_size = 50
train_dataloader = tc.utils.data.DataLoader(training_data, batch_size=batch_size)
test_dataloader = tc.utils.data.DataLoader(test_data, batch_size=batch_size)

# Get cpu or gpu device for training.
device = "cuda" if tc.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# The Conv1d-Maxpool1d model from Yoon Kim, 2014 - 'Convolutional Neural Networks for Text Classification'.
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

runner = Runner(max_epochs=1, verbose=True)
runner.run(model, train_dataloader, test_dataloader, device, criterion, optimizer)
print("Done!")


tokenizer = training_data.tokenizer
vocab = training_data.vocab
n_tokens = 400
hypothetical_review = "The movie was good, but there were some over the top moments especially in the action scenes. Overall B+."
vectors = np.stack([vocab.vectors[vocab.stoi[token]] for token in tokenizer(hypothetical_review)], axis=0)
vectors = vectors[0:n_tokens]
pad_len = n_tokens-len(vectors)
pad_shape = (pad_len, 300)
if pad_len > 0:
    vectors = np.concatenate([vectors, np.zeros(dtype=np.float32, shape=pad_shape)], axis=0)
vectors = np.transpose(vectors) # convert to NCH format.
vectors = tc.from_numpy(vectors)
model.eval()
logits = model.forward(tc.unsqueeze(vectors, dim=0))
probs = tc.nn.Softmax(dim=-1)(logits)
print(hypothetical_review)
print(probs)