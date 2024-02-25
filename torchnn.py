# pytorch neural networks

from torch import nn, save, load
# optimizer
from torch.optim import Adam
# to load datasets
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose

# Used to load model after initial training
import torch
from PIL import Image
import os, sys, datetime

# image classifier neural network
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = (3,3)),
            nn.ReLU(),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3,3)),
            nn.ReLU(),            
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3,3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features = 64*(28-6)*(28-6), out_features =  62)  # Changed output layer size to 62 for EMNIST
        )    

    def forward(self, x):
        return self.model(x)

def train_nn():
    # import dataset
    train = datasets.EMNIST(root = "data", download=True, train=True, transform=ToTensor(), split='letters')
    dataset = DataLoader(dataset=train, batch_size=32) # 62 classes: digits, uppercase and lowercase letters

    # Local CPU training
    print("\n\nBeginning the training process...")
    for epoch in range(2):
        print(f"Current time: {datetime.datetime.now().time()}")

        for batch in dataset:
            X, y = batch
            X, y = X.to('cpu'), y.to('cpu')
            # Make prediction
            yhat = clf(X)
            loss = loss_fn(yhat, y)

            # Apply backpropagation
            opt.zero_grad() # Zero out gradient
            loss.backward() # Calculate gradients
            opt.step()

        print(f"Epoch: {epoch} loss is {loss.item()}")


    with open('model_state.pt', 'wb') as f:
        torch.save(clf.state_dict(), f)

# Training function
if __name__ == "__main__":
    # Create instance of the neural network, loss, optimizer
    clf = ImageClassifier().to('cpu')
    opt = Adam(clf.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # Train the model if the state doesn't exist:
    retrain_model = False
    if retrain_model:
        train_nn()

    # Load the stored PyTorch NN
    with open('model_state.pt', 'rb') as f:
        clf.load_state_dict(torch.load(f))

    # Load the RGB image
    if len(sys.argv) > 1:
        img = Image.open(f"data/{sys.argv[1]}.jpg")
    else:
        # Default to image 1 if no arg is given
        img = Image.open('data/img_1.jpg')

    # Convert the image to grayscale
    img = img.convert(mode='L')
    # Resize the image to the appropriate size (based on the MNIST training set)
    img = img.resize((28, 28))

    img_tensor = ToTensor()(img).unsqueeze(0).to('cpu')

    print(torch.argmax(clf(img_tensor)))
