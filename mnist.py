import torch
import torchvision

import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Training parameters
n_epochs = 250
batch_size_train = 128
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=True, download=True, 
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.1307,), (0.3081,))
        ])),
    batch_size=batch_size_train, shuffle=True, num_workers=5)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=False, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.1307,), (0.3081,))
        ])),
    batch_size=batch_size_test, shuffle=True)
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

print(example_data.shape)
print(example_targets.shape)
print(example_targets[0:9])

# fig = plt.figure()
# for i in range(6):
#   plt.subplot(2,3,i+1)
#   plt.tight_layout()
#   plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
#   plt.title("Ground Truth: {}".format(example_targets[i]))
#   plt.xticks([])
#   plt.yticks([])
# plt.show()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        neurons = 250
        self.fc1 = nn.Linear(28*28, neurons)
        self.fc2 = nn.Linear(neurons, neurons)
        self.fc3 = nn.Linear(neurons, 10)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        x = x.view(-1, 28*28)  # Flatten the input
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x

network = Net().to(device)
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)


def train(epoch):
    total_loss = 0
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ' \
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            
    return total_loss / len(train_loader.dataset)

def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = network(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ' \
          f'({100. * correct / len(test_loader.dataset):.0f}%)\n')
    return test_loss

train_losses = []
test_losses = []
for epoch in range(1, n_epochs + 1):
    loss = train(epoch)
    train_losses.append(loss)
    loss = test()
    test_losses.append(loss)

print("Training complete")

# Plot the loss over the epochs in two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot training loss
ax1.plot(train_losses, label='Train Loss')
ax1.set_title('Training Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()

# Plot test loss
ax2.plot(test_losses, label='Test Loss')
ax2.set_title('Test Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()


plt.tight_layout()
plt.show()

# Get a batch of test data
examples = enumerate(test_loader)
_, (example_data, example_targets) = next(examples)

# Get predictions from the trained model
network.eval()
with torch.no_grad():
    example_data, example_targets = example_data.to(device), example_targets.to(device)
    predictions = network(example_data[:10])
    predicted_labels = predictions.argmax(dim=1)

# Plot the images with their predicted labels
fig = plt.figure()
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0].cpu(), cmap='gray', interpolation='none')
    plt.title(f"Pred: {predicted_labels[i].item()} Actual: {example_targets[i].item()}")
    plt.xticks([])
    plt.yticks([])
plt.show()