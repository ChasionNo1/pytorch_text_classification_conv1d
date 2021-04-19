import torch as tc
from datasets import IMDBDatasetEmbedded
from classifier import Conv1dTextClassifier
from runner import Runner

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

runner = Runner(max_epochs=100, verbose=True)
runner.run(model, train_dataloader, test_dataloader, device, criterion, optimizer)
print("Done!")

