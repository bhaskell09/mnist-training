import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from net import Net
from sty import fg

# Verify matplotlib backend is TkAgg
backend = matplotlib.get_backend()
assert backend == "tkagg", f"Backend is {backend}, not TkAgg :("

# Training parameters
n_epochs = 5
batch_size_train = 512
batch_size_test = 1000
learning_rate = 0.0001
momentum = 0.5
log_interval = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print("Using GPU for training")
else:
    print(fg.yellow + "WARNING: Using CPU for training" + fg.rs)
    response = input(fg.yellow + "Do you want to continue training on CPU? (May result in slower training) (y/n): " + fg.rs).strip().lower()
    if response != 'y':
        print("Exiting...")
        exit()

# Load MNIST dataset
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=True, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0,), (1,))
        ])),
    batch_size=batch_size_train, shuffle=True, num_workers=15)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=False, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0,), (1,))
        ])),
    batch_size=batch_size_test, shuffle=True)


network = Net().to(device)
optimizer = optim.Adam(network.parameters(), lr=learning_rate)

# Training function


def train(epoch):
    network.train()
    total_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    train_accuracy = 100. * correct / len(train_loader.dataset)

    average_train_loss = total_loss / len(train_loader.dataset)

    print(f'Train set: Average loss: {average_train_loss:.4f}, Accuracy: {correct}/{len(train_loader.dataset)} '
          f'({train_accuracy:.2f}%)')
    return average_train_loss, train_accuracy

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

    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({test_accuracy:.2f}%)')

    return test_loss, test_accuracy


# Training loop
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []
best_test_loss = float('inf')
no_improvement_counter = 0
for epoch in range(1, n_epochs + 1):
    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test()

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

    if test_loss < best_test_loss:
        best_test_loss = test_loss
        no_improvement_counter = 0
        torch.save(network.state_dict(), "mnist_model_Conv.pth")
    else:
        no_improvement_counter += 1
        print(
            f"No improvement in test loss for {no_improvement_counter} epochs.")
        if no_improvement_counter >= 10:
            print("Stopping early due to no improvement in test loss.")
            break
print("Training complete")

# Plot loss over epochs
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(train_losses, label='Train Loss')
ax1.set_title('Training Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()

ax2.plot(test_losses, label='Test Loss')
ax2.set_title('Test Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()

plt.tight_layout()
plt.show()

# Plot accuracy over epochs
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(train_accuracies, label='Train Accuracy')
ax1.set_ylim([0, 100])
ax1.set_title('Training Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy (%)')
ax1.legend()

ax2.plot(test_accuracies, label='Test Accuracy')
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

examples = enumerate(test_loader)
_, (example_data, example_targets) = next(examples)

network.eval()
with torch.no_grad():
    example_data, example_targets = example_data.to(device), example_targets.to(device)
    output = network(example_data[:10])
    # Get predicted class
    predicted_labels = output.argmax(dim=1)
    # Get confidence scores directly from the network output
    # Since the network already applies softmax in its forward method
    confidence_scores = output[range(len(predicted_labels)), predicted_labels]
    
fig = plt.figure(figsize=(12, 6))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0].cpu(), cmap='gray', interpolation='none')
    # Display prediction, actual label, and confidence percentage
    plt.title(
        f"Pred: {predicted_labels[i].item()}\nActual: {example_targets[i].item()}\nConf: {confidence_scores[i].item()*100:.1f}%")
    plt.xticks([])
    plt.yticks([])
plt.show()

y_pred = []
y_true = []

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = network(data)
        pred = output.argmax(dim=1, keepdim=True)
        y_pred.extend(pred.view(-1).cpu().numpy())
        y_true.extend(target.cpu().numpy())

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Convert confusion matrix to percentages
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sn.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='Blues',
           xticklabels=range(len(cm)), yticklabels=range(len(cm)))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix (Percentages)')
plt.show()

# Append the accuracy information to accuracy.txt
with open('accuracy.txt', 'a') as f:
    # Get model name from the saved path
    model_name = "mnist_model_Conv.pth"
    # Extract activation function from network architecture
    activation_func = "relu"  # This is hardcoded based on your Net class using F.relu
    # Write the accuracy to the file
    f.write(f"{activation_func} accuracy: {test_accuracies[-1]:.2f}%\n")