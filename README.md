# :evergreen_tree: The Self Organizing Tree

<img src="https://github.com/LumRamabaja/Self-Organizing-Tree/blob/main/img/conv_sot.png" class="center">

The self organizing tree (SOT) is a type of artificial neural network based on the self organizing map ([SOM](https://en.wikipedia.org/wiki/Self-organizing_map)). It uses an unsupervised, competitive learning algorithm to store representations of the input space. Unlike the conventional SOM, where a data point is given to every node of the SOM, in SOTs a forward propagation requires to compute a competition between only $log(N)$ nodes, where $N$ is the number of nodes the model has. This makes the SOT computationally significantly more efficient than a conventional SOM. The neighbourhood of a SOT is also differently defined than in a SOM. While in a 2D SOM, nodes are organized in a grid like fashion, the SOT's nodes are organized in the form of a binary tree. The binary tree structure allows SOTs to find a best matching unit with fewer steps, however, its computational efficiency comes with a cost. While in the SOM, $N$ nodes can become best matching units, in the SOT only $N/2$ of the nodes can potentially become best matching units. If you want to know more about the self organizng tree, checkout this [blog post](https://lums.blog/self-organizing-tree)

## Example
```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from Conv2dSOT import Conv2dSOT
import torch

transform = transforms.ToTensor()

training_set = datasets.FashionMNIST("~/.pytorch/F_MNIST_data", download = False, train = True, transform = transform)
testing_set = datasets.FashionMNIST("~/.pytorch/F_MNIST_data", download = False, train = False, transform = transform)

train_dataloader = DataLoader(training_set,  
                            shuffle = True,  
                            batch_size = 1,  
                            num_workers = 0  
                            )
test_dataloader = DataLoader(testing_set,  
                            shuffle = True,  
                            batch_size = 1,  
                            num_workers = 0  
                            )


# initialize convolutional self organizing tree
device = torch.device('cpu')
tree = Conv2dSOT(number_of_leaves = 20,
                kernel_size = 5,
                stride = 2,
                padding = 0,
                lr = 0.2,
                device = device
               )

iterations = 10000
num = 0
for data in train_dataloader:
    x, y = data
    bmu = tree.forward(x.reshape(28,28) )
    num += 1
    if num >= iterations:
        break
```
