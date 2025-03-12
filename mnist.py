import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib

# Verify matplotlib backend is TkAgg
backend = matplotlib.get_backend()
assert backend == "tkagg", f"Backend is {backend}, not TkAgg :("

# Training parameters
n_epochs = 500
batch_size_train = 256
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MNIST dataset
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=True, download=True, 
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])),
    batch_size=batch_size_train, shuffle=True, num_workers=2)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=False, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])),
    batch_size=batch_size_test, shuffle=False)

# Define Neural Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        neurons = 50
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

# Training function
def train(epoch):
    network.train()
    total_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    train_accuracy = 100. * correct / len(train_loader.dataset)
    return total_loss / len(train_loader.dataset), train_accuracy

# Testing function
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
    test_accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({test_accuracy:.2f}%)\n')
    
    return test_loss, test_accuracy

# Training loop
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(1, n_epochs + 1):
    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test()
    
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

print("Training complete")

# Plot loss over epochs
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(range(1, n_epochs + 1), train_losses, label='Train Loss')
ax1.set_title('Training Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()

ax2.plot(range(1, n_epochs + 1), test_losses, label='Test Loss')
ax2.set_title('Test Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()

plt.tight_layout()
plt.show()

# Plot accuracy over epochs
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(range(1, n_epochs + 1), train_accuracies, label='Train Accuracy')
ax1.set_ylim([0, 100])
ax1.set_title('Training Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy (%)')
ax1.legend()

ax2.plot(range(1, n_epochs + 1), test_accuracies, label='Test Accuracy')
ax2.set_ylim([0, 100])
ax2.set_title('Test Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.legend()

plt.tight_layout()
plt.show()

# Get a batch of test data and visualize predictions
examples = enumerate(test_loader)
_, (example_data, example_targets) = next(examples)

network.eval()
with torch.no_grad():
    example_data, example_targets = example_data.to(device), example_targets.to(device)
    predictions = network(example_data[:10])
    predicted_labels = predictions.argmax(dim=1)

fig = plt.figure()
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0].cpu(), cmap='gray', interpolation='none')
    plt.title(f"Pred: {predicted_labels[i].item()} Actual: {example_targets[i].item()}")
    plt.xticks([])
    plt.yticks([])
plt.show()
