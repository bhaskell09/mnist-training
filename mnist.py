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
import os
import datetime

# Verify matplotlib backend is TkAgg
backend = matplotlib.get_backend()
assert backend == "tkagg", f"Backend is {backend}, not TkAgg :("

# Training parameters
n_epochs = 500
batch_size_train = 512
batch_size_test = 1000
learning_rate = 0.001  # Slightly higher initial learning rate
momentum = 0.5
log_interval = 10

# Model identifier
model_name = "mnist_model_Conv.pth"
activation_function = "ReLU"  # The activation function used in the network

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

# Add learning rate scheduler
# ReduceLROnPlateau reduces learning rate when a metric stops improving
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',           # Monitor loss (minimize)
    factor=0.5,           # Multiply learning rate by this factor when reducing
    patience=5,           # Number of epochs with no improvement after which LR will be reduced
    verbose=True,         # Print message when LR is reduced
    min_lr=1e-6           # Lower bound on the learning rate
)

# Function to get current learning rate
def get_lr():
    for param_group in optimizer.param_groups:
        return param_group['lr']

# Function to save accuracy information to a file
def save_accuracy_info(filename, activation, accuracy, epoch_count):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create the file if it doesn't exist, otherwise append to it
    mode = 'a' if os.path.exists('accuracy.txt') else 'w'
    
    with open('accuracy.txt', mode) as f:
        if mode == 'w':
            f.write("Timestamp, Model Filename, Activation Function, Test Accuracy, Epochs Trained\n")
        f.write(f"{timestamp}, {filename}, {activation}, {accuracy:.2f}%, {epoch_count}\n")
    
    print(f"Accuracy information saved to accuracy.txt")

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
learning_rates = []  # Track learning rates
best_test_loss = float('inf')
best_test_accuracy = 0.0
no_improvement_counter = 0

# Print initial learning rate
current_lr = get_lr()
print(f"Initial learning rate: {current_lr:.6f}")

for epoch in range(1, n_epochs + 1):
    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test()
    
    # Get and print current learning rate after train/test
    current_lr = get_lr()
    print(f"Current learning rate: {current_lr:.6f}")
    learning_rates.append(current_lr)
    
    # Step the scheduler based on validation loss
    scheduler.step(test_loss)

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

    # Track best accuracy
    if test_acc > best_test_accuracy:
        best_test_accuracy = test_acc

    if test_loss < best_test_loss:
        best_test_loss = test_loss
        no_improvement_counter = 0
        torch.save(network.state_dict(), model_name)
    else:
        no_improvement_counter += 1
        print(
            f"No improvement in test loss for {no_improvement_counter} epochs.")
        if no_improvement_counter >= 10:
            print("Stopping early due to no improvement in test loss.")
            break

# Save accuracy information at the end of training
save_accuracy_info(model_name, activation_function, best_test_accuracy, epoch)

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

# Plot learning rate over epochs
plt.figure(figsize=(10, 5))
plt.plot(learning_rates)
plt.title('Learning Rate Schedule')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.yscale('log')  # Log scale often better for visualizing learning rates
plt.grid(True)
plt.show()

# Get a batch of test data and visualize predictions
examples = enumerate(test_loader)
_, (example_data, example_targets) = next(examples)

network.eval()
with torch.no_grad():
    example_data, example_targets = example_data.to(device), example_targets.to(device)
    output = network(example_data[:10])
    predicted_labels = output.argmax(dim=1)
    
fig = plt.figure()
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0].cpu(), cmap='gray', interpolation='none')
    plt.title(
        f"Pred: {predicted_labels[i].item()} Actual: {example_targets[i].item()}")
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