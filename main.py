import torch as tc
import torchtext as tt
from utils import get_tokenizer, get_vectors, text_pipeline, label_pipeline, ProcessedIterableDataset
from functools import partial
from classifier import Conv1dTextClassifier
from runner import Runner

# Datasets.
training_data = tt.datasets.IMDB(root='data', split='train')
test_data = tt.datasets.IMDB(root='data', split='test')

# Preprocessing.
tokenizer = get_tokenizer()
vectors = get_vectors()
text_preprocessing = partial(text_pipeline, tokenizer=tokenizer, vectors=vectors)
dataset_map = lambda y,x: (text_preprocessing(x), label_pipeline(y))
training_data = ProcessedIterableDataset(training_data, dataset_map)
test_data = ProcessedIterableDataset(test_data, dataset_map)

# Dataloaders.
batch_size = 50
train_dataloader = tc.utils.data.DataLoader(training_data, batch_size=batch_size)
test_dataloader = tc.utils.data.DataLoader(test_data, batch_size=batch_size)

# Device.
device = "cuda" if tc.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Model from Yoon Kim, 2014 - 'Convolutional Neural Networks for Sentence Classification'.
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

# Runner.
runner = Runner(max_epochs=1, verbose=True)
runner.run(model, train_dataloader, test_dataloader, device, criterion, optimizer)
print("All done!")

# Demo.
model.eval()
hypothetical_review = "The movie was good, but there were some over the top moments especially in the action scenes. Overall B+."
x = tc.from_numpy(text_preprocessing(hypothetical_review))
y_logits = model.forward(tc.unsqueeze(x, dim=0))
y_probs = tc.nn.Softmax(dim=-1)(y_logits)
print(hypothetical_review)
print(y_probs)