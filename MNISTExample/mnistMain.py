from torchvision import datasets, transforms
import torch

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load MNIST data
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Convert the dataset into a DataLoader to easily iterate over the data
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Initialize empty lists for images and labels
image_list = []
label_list = []

# Iterate through the DataLoader to collect individual images and labels
for images, labels in trainloader:
    for img, label in zip(images, labels):
        # Flatten the image to a 1D vector (28x28=784)
        img = img.view(-1).cpu().detach().numpy().tolist()  # Convert image tensor to a list
        label = label.cpu().detach().numpy().tolist()  # Convert label tensor to a list
        
        # Append the image and label to the lists
        image_list.append(img)
        label_list.append(label)
        
labels = []
for i in range(len(label_list)):
    l = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,]
    l[label_list[i]] = 1
    labels.append(l)

from neuralLibrary.main import Network

n = Network()
n.addLayer(784, "relu")
n.addLayer(64, "relu")
n.addLayer(10, "softmax")
n.createWeightsAndBiases()
n.train(image_list, labels, 1)