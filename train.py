from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import torch
from torch.utils.data import DataLoader
from torch import nn


transform = T.Compose([
    T.Resize((224, 224)),  # Resize to fit the input dimensions of the network
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# root/{classX}/x001.jpg

tiny_imagenet_dataset_train = ImageFolder(root='tiny-imagenet/tiny-imagenet-200/train', transform=transform)
tiny_imagenet_dataset_val = ImageFolder(root='tiny-imagenet/tiny-imagenet-200/val', transform=transform)

# Define the custom neural network
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        # Define layers of the neural network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # Add more layers...

        self.fc1 = nn.Linear(128*56*56, 200) # 200 is the number of classes in TinyImageNet

    def forward(self, x):
        # Define forward pass
        x = self.pool(self.conv1(x).relu())
        x = self.pool(self.conv2(x).relu())
        x = torch.flatten(x,1)
        x = self.fc1(x)
        # B x 3 x 224 x 224
        return x
    
def train(epoch, model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()

        # todo...done?
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')

    # Validation loop
def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0

    correct, total = 0, 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()

            # todo...
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100. * correct / total

    print(f'Validation Loss: {val_loss:.6f} Acc: {val_accuracy:.2f}%')
    return val_accuracy

def main():
    model = CustomNet().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    best_acc = 0

    # Run the training process for {num_epochs} epochs
    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        train(epoch, model, CustomNet.train_loader, criterion, optimizer)

        # At the end of each training iteration, perform a validation step
        val_accuracy = validate(model, CustomNet.val_loader, criterion)

        # Best validation accuracy
        best_acc = max(best_acc, val_accuracy)


    print(f'Best validation accuracy: {best_acc:.2f}%')

if __name__ == "__main__":
    main()