import time
import os
import torch.nn as nn
import sys
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models

def main(argv):
    start = time.time()

    if len(sys.argv) < 2:
        print("Error: Missing argument - test dataset")
        exit(0)

    test_dir = sys.argv[1]
    
    batch_size = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])
        ])

    #classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    #start1 = time.time()
    test_dataset = datasets.ImageFolder(root=test_dir,transform=data_transform)
    test_dataset_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size, shuffle=True)
    example = next(iter(test_dataset_loader))[0].to(device)
    #print('Loading data : %s seconds' % (time.time() - start1))

    #start2 = time.time()
    model = torchvision.models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    model.eval()
    model.to(device)
    #print('Loading Model : %s seconds' % (time.time() - start2))

    #torch.cuda.synchronize()
    #start3 = time.time()
    with torch.no_grad():
        output = model(example)
    #torch.cuda.synchronize()
    #print('Inference : %s seconds' % (time.time() - start3))
    _, predicted = torch.max(output.data, 1)
    #print(predicted)

    #print('Total : %s seconds' % (time.time() - start))
    #print('-'*30)



if __name__ == "__main__":
    main(sys.argv[1:])
