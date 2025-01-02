import numpy as np
np.bool = np.bool_
import mxnet as mx
from mxnet import gluon, autograd, nd
from mxnet.gluon import nn, data as gdata
import matplotlib.pyplot as plt
import numpy as np
import time
import os

# ---------------------------
# Setup Context and Config
# ---------------------------
try:
    num_gpus = mx.context.num_gpus() if hasattr(mx.context, 'num_gpus') else 0
    if num_gpus > 0:
        ctx = [mx.gpu(i) for i in range(num_gpus)]
    else:
        ctx = [mx.cpu()]
except AttributeError:
    ctx = [mx.cpu()]
    print("GPU context not available, falling back to CPU.")
    
print(f"Using Context: {ctx}")

# ---------------------------
# Step 2: Dataset Preparation
# ---------------------------
batch_size = 64
transformer = gdata.vision.transforms.Compose([
    gdata.vision.transforms.ToTensor(),
    gdata.vision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Fixed Normalize
])

train_data = gdata.DataLoader(
    gdata.vision.CIFAR10(train=True).transform_first(transformer),
    batch_size=batch_size, shuffle=True, num_workers=4)

test_data = gdata.DataLoader(
    gdata.vision.CIFAR10(train=False).transform_first(transformer),
    batch_size=batch_size, shuffle=False, num_workers=4)

# ---------------------------
# Step 3: Define the Model
# ---------------------------
class CIFAR10Model(nn.Block):
    def __init__(self):
        super(CIFAR10Model, self).__init__()
        self.conv1 = nn.Conv2D(32, kernel_size=3, activation='relu')
        self.pool1 = nn.MaxPool2D(pool_size=2)
        self.conv2 = nn.Conv2D(64, kernel_size=3, activation='relu')
        self.pool2 = nn.MaxPool2D(pool_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(128, activation='relu')
        self.fc2 = nn.Dense(10)

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.flatten(x)
        x = self.fc1(x)
        return self.fc2(x)

net = CIFAR10Model()
net.initialize(ctx=ctx)

# ---------------------------
# Step 4: Loss and Optimizer
# ---------------------------
loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 0.001})

# ---------------------------
# Step 5: Training with Performance Metrics
# ---------------------------
def train_model(net, train_data, test_data, loss_fn, trainer, ctx, epochs=5):
    training_times = []
    train_accuracies, test_accuracies = [], []
    train_losses = []  # List to store the training loss for each epoch
    
    for epoch in range(epochs):
        start_time = time.time()
        train_loss, train_acc, test_acc = 0.0, 0.0, 0.0
        
        for data, label in train_data:
            data = data.as_in_context(ctx[0])
            label = label.as_in_context(ctx[0]).astype('float32')  # Convert label to float32
            with autograd.record():
                output = net(data)
                loss = loss_fn(output, label)
            loss.backward()
            trainer.step(batch_size)
            train_loss += loss.mean().asscalar()
            train_acc += (output.argmax(axis=1) == label).mean().asscalar()
        
        for data, label in test_data:
            data = data.as_in_context(ctx[0])
            label = label.as_in_context(ctx[0]).astype('float32')  # Convert label to float32
            output = net(data)
            test_acc += (output.argmax(axis=1) == label).mean().asscalar()
        
        end_time = time.time()
        epoch_time = end_time - start_time
        training_times.append(epoch_time)
        train_accuracies.append(train_acc / len(train_data))
        test_accuracies.append(test_acc / len(test_data))
        train_losses.append(train_loss / len(train_data))
        
        print(f"Epoch {epoch+1}, Time: {epoch_time:.2f}s, Loss: {train_loss/len(train_data):.4f}, "
              f"Train Acc: {train_accuracies[-1]:.4f}, Test Acc: {test_accuracies[-1]:.4f}")
    
    print("\nAverage Training Time per Epoch: {:.2f}s".format(np.mean(training_times)))
    print("Final Training Accuracy: {:.4f}".format(train_accuracies[-1]))
    print("Final Testing Accuracy: {:.4f}".format(test_accuracies[-1]))
    
    # Plotting the training and test accuracy and loss
    plt.figure(figsize=(12, 4))

    # Plot Training & Testing Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), train_accuracies, label='Train Accuracy')
    plt.plot(range(epochs), test_accuracies, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Test Accuracy')

    # Plot Training Loss
    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), train_losses, label='Train Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss')

    plt.tight_layout()
    plt.show()

# ---------------------------
# Step 6: Fault Tolerance
# ---------------------------
checkpoint_file = 'cifar10_checkpoint.params'
def save_checkpoint(net):
    net.save_parameters(checkpoint_file)
    print("Checkpoint saved.")

def load_checkpoint(net):
    if os.path.exists(checkpoint_file):
        net.load_parameters(checkpoint_file, ctx=ctx)
        print("Checkpoint loaded successfully.")
    else:
        print("No checkpoint found.")

try:
    print("\nTraining started...")
    train_model(net, train_data, test_data, loss_fn, trainer, ctx, epochs=5)
    save_checkpoint(net)
except KeyboardInterrupt:
    print("Training interrupted. Loading from last checkpoint...")
    load_checkpoint(net)

# ---------------------------
# Step 7: Scalability Test
# ---------------------------
def scalability_test(batch_sizes):
    for b_size in batch_sizes:
        print(f"\nTesting Scalability with Batch Size: {b_size}")
        scaled_train_data = gdata.DataLoader(
            gdata.vision.CIFAR10(train=True).transform_first(transformer),
            batch_size=b_size, shuffle=True)
        
        start_time = time.time()
        for data, label in scaled_train_data:
            data = data.as_in_context(ctx[0])
            label = label.as_in_context(ctx[0]).astype('float32')  # Convert label to float32
            with autograd.record():
                output = net(data)
                loss = loss_fn(output, label)
            loss.backward()
            trainer.step(b_size)
        end_time = time.time()
        
        print(f"Batch Size: {b_size}, Training Time: {end_time - start_time:.2f}s")

scalability_test([32, 64, 128])

# ---------------------------
# Step 8: Save Final Model
# ---------------------------
net.save_parameters('final_cifar10_model.params')
print("\nFinal model saved.")
