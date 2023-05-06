import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def load_cifar10(batch_size):
    transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = datasets.CIFAR10(root='./data',
                                train=True,
                                download=True,
                                transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True)

    testset = datasets.CIFAR10(root='./data',
                               train=False,
                               download=True,
                               transform=transform)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader
