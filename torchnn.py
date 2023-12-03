# pytorch neural networks
from torch import nn, save, load
# optimizer
from torch.optim import Adam
# to load datsets
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose

# Used to load model after initial training
import torch
from PIL import Image
import os, sys

# import dataset
train = datasets.MNIST(root = "data", download=True, train=True, transform=ToTensor())
dataset = DataLoader(dataset=train, batch_size=32)
# 10 classes, one for each digit 0 through 9


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
            nn.Linear(in_features = 64*(28-6)*(28-6), 
                      out_features =  10)
        )    

    def forward(self, x):
        return self.model(x)

# instance of the neural network, loss, optimizer
clf = ImageClassifier().to('cpu')
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

def train_nn():
    # local cpu training ~ 15 minutes
    print("\n\nBeginning the training process...")
    for epoch in range(10):
        for batch in dataset:
            X,y = batch
            X,y = X.to('cpu'), y.to('cpu')
            # make prediction
            yhat = clf(X)
            loss = loss_fn(yhat, y)

            # Apply backpro
            opt.zero_grad() # zero out gradient
            loss.backward() # calculate gradients
            opt.step()

        print(f"Epoch: {epoch} loss is {loss.item()}")

    with open('model_state.pt', 'wb') as f:
        torch.save(clf.state_dict(), f)

# training function
if __name__ == "__main__":
    # train the model if the state doesn't exist:
    if not(os.path.exists('model_state.pt')):
        train_nn()

    # load the stored pytorch nn
    with open('model_state.pt', 'rb') as f:
        clf.load_state_dict(torch.load(f))

    # Load the RGB image
    if len(sys.argv) > 1:
        img = Image.open(f"{sys.argv[1]}.jpg")
    else:
        # default to image 1 if no arg is given
        img = Image.open('img_1.jpg')

    # Convert the image to grayscale
    img = img.convert(mode='L')
    # resize the image to the appropriate size (based on the MNIST training set)
    img = img.resize((28, 28))

    # img = Image.open('img_9.jpg')
    # transform = Compose([ToTensor(),])

    img_tensor = ToTensor()(img).unsqueeze(0).to('cpu')

    print(torch.argmax(clf(img_tensor)))
