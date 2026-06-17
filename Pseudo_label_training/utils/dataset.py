import json
import torch.utils.data as data
import torch
import random
import numpy as np
from torch.utils.data import random_split

import torchvision
from torchvision import transforms
from torchvision import datasets as dset
npy_file_path_train = np.load("/home/gargatik/extra2/block-selection-for-OOD-detection/figure/res18/cifar10_resnet18_Kmeans20_train_MBT.npy") #change the pseuso label path

npy_file_path_test  = np.load("/home/gargatik/extra2/block-selection-for-OOD-detection/figure/res18/cifar10_resnet18_Kmeans20_test_MBT.npy") #change the pseuso label path
mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)

size = 32
# for the supervised benchmarking on all ID datsets
train_transform_cifar = transforms.Compose([transforms.Resize([size,size]), transforms.RandomHorizontalFlip(), transforms.RandomCrop(size, padding=4),
                               transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

test_transform_cifar = transforms.Compose([transforms.Resize([size,size]), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])#, )
test_transform_cifar_vflip = transforms.Compose([transforms.Resize([size,size]), transforms.RandomVerticalFlip(p=1.0)  ,transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])#, )
test_transform_cifar_hflip = transforms.Compose([transforms.Resize([size,size]), transforms.RandomHorizontalFlip(p=1.0),transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])#, )


# for the unsupervised benchmarking on all ID color datsets

# train_transform_cifar = transforms.Compose([transforms.Resize([size,size]),  transforms.RandomCrop(size, padding=4),
#                                transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
# test_transform_cifar = transforms.Compose([transforms.Resize([size,size]), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])#, )
# test_transform_cifar_vflip = transforms.Compose([transforms.Resize([size,size]), transforms.RandomVerticalFlip(p=1.0)  ,transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])#, )
# test_transform_cifar_hflip = transforms.Compose([transforms.Resize([size,size]), transforms.RandomHorizontalFlip(p=1.0),transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])#, )



# for the unsupervised benchmarking on all ID grayscale datsets


# train_transform_cifar = transforms.Compose([transforms.Resize([size,size]),transforms.Grayscale(num_output_channels=3), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])#, )# only for mnist
# test_transform_cifar = transforms.Compose([transforms.Resize([size,size]),transforms.Grayscale(num_output_channels=3), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])#, )
# test_transform_cifar_vflip = transforms.Compose([transforms.Resize([size,size]),transforms.Grayscale(num_output_channels=3), transforms.RandomVerticalFlip(p=1.0)  ,transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])#, )
# test_transform_cifar_hflip = transforms.Compose([transforms.Resize([size,size]),transforms.Grayscale(num_output_channels=3), transforms.RandomHorizontalFlip(p=1.0),transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])#, )




class jigsaw_dataset(data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        x, y = self.dataset[index]
        
        s = int(float(x.size(1)) / 3)
        
        
        x_ = torch.zeros_like(x)
        tiles_order = random.sample(range(9), 9)
        for o, tile in enumerate(tiles_order):
            i = int(o/3)
            j = int(o%3)
            
            ti = int(tile/3)
            tj = int(tile%3)
            # print(i, j, ti, tj)
            x_[:, i*s:(i+1)*s, j*s:(j+1)*s] = x[:, ti*s:(ti+1)*s, tj*s:(tj+1)*s] 
        return x_, y
        
def get_cifar_jigsaw(dataset, folder, batch_size, test=False):
    test_transform_cifar = transforms.Compose([transforms.Resize([size,size]), transforms.ToTensor()])
    train_ = not test
    
    if dataset == 'cifar10':
        train_data = dset.CIFAR10(folder, train=train_, transform=test_transform_cifar, download=True)

        test_data = dset.CIFAR10(folder, train=train_, transform=test_transform_cifar, download=True)
    if dataset == 'cifar100':
        train_data = dset.CIFAR10(folder, train=train_, transform=test_transform_cifar, download=True)

        test_data = dset.CIFAR10(folder, train=train_, transform=test_transform_cifar, download=True)
    if dataset == 'svhn':
        train_data = dset.SVHN(folder, split='train', transform=test_transform_cifar , download=True)
        train_data = dset.SVHN(folder, split='est', transform=test_transform_cifar , download=True)

    if dataset == 'fmnist':
        train_data = dset.FashionMNIST(folder, train=True, transform=test_transform_cifar , download=True)
        train_data = dset.FashionMNIST(folder, train=False, transform=test_transform_cifar , download=True)

    if dataset == 'celeba':
        train_data = dset.CelebA(root='./data/wbandar1/datasets', split='train', \
                                            transform=test_transform_cifar, download=True)
        test_data = dset.CelebA(root='./data/wbandar1/datasets', split='test', \
                                            transform=test_transform_cifar, download=True)

    

    jigsaw = jigsaw_dataset(test_data)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=False, pin_memory=True, num_workers = 4)
    jigsaw_loader = torch.utils.data.DataLoader(jigsaw, batch_size, shuffle=False, pin_memory=True, num_workers = 4)

    print(len(test_data), "length=len(test_data)")
    print(len(train_data), "length=len(train_data)")
    
    return train_loader, jigsaw_loader

def read_conf(json_path):
    """
    read json and return the configure as dictionary.
    """
    with open(json_path) as json_file:
        config = json.load(json_file)
    return config


#for cifar training

def get_cifar(dataset, folder, batch_size, eval=False):
    if eval==True:
        train_transform_cifar_ = test_transform_cifar
    else:
        train_transform_cifar_ = train_transform_cifar
    if dataset == 'cifar10':
        train_data = dset.CIFAR10(folder, train=True, transform=train_transform_cifar_, download=True)
        # train_data = dset.CIFAR100(folder, train=True, transform=CifarPairTransform(train_transform = True), download=True)
        are_labels_same = np.array_equal(np.array(train_data.targets), npy_file_path_train)
         

        if are_labels_same:
            
            print("Labels from CIFAR10 training dataset and ground truth labels are the same.")
        else:
            # train_data.targets =npy_file_path_train
            print("Labels from CIFAR10 training dataset and ground truth labels are different.")
            print(dataset,"name of dataset")

        test_data = dset.CIFAR10(folder, train=False, transform=test_transform_cifar, download=True)
        are_labels_same = np.array_equal(np.array(test_data.targets), npy_file_path_test)
         

        if are_labels_same:
            
            print("Labels from CIFAR10 train dataset and ground truth labels are the same.")
        else:
            # test_data.targets =npy_file_path_test
            print("Labels from CIFAR10 train dataset and ground truth labels are different.")
        num_classes = 10

    # Calculate split sizes (95% training, 5% validation)
    train_size = int(0.95 * len(train_data))
    valid_size = len(train_data) - train_size

    # Split the dataset into training and validation sets
    train_data, valid_data = random_split(train_data, [train_size, valid_size])
        
    # DataLoaders for training and validation
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False, pin_memory=True, num_workers = 4)
    
    return train_loader, valid_loader

#for fmnist ID training 
def get_fmnist_t( batch_size):
    folder = '/home/gargatik/extra2/block-selection-for-OOD-detection/data/wbandar1/datasets/'
    train_data = dset.FashionMNIST(folder, train=True, transform=train_transform_cifar , download=True)
    
    test_data = dset.FashionMNIST(folder, train=False, transform=test_transform_cifar, download=True)
    are_labels_same = np.array_equal(np.array(train_data.targets), npy_file_path_train)
    if are_labels_same:
            
        print("Labels from fmnist train dataset and ground truth labels are the same.")
    else:
        # test_data.targets =npy_file_path_test
        print("Labels from fmnist train dataset and ground truth labels are different.")

    train_size = int(0.95 * len(train_data))
    valid_size = len(train_data) - train_size

    # Split the dataset into training and validation sets
    train_data, valid_data = random_split(train_data, [train_size, valid_size])


    train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True, pin_memory=True, num_workers = 4) 
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size, shuffle=False, pin_memory=True, num_workers = 4) 
    test_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False, pin_memory=True, num_workers = 4)    
    return train_loader, valid_loader

#for svhn ID training 
def get_svhn_t( batch_size):
    folder = '/home/gargatik/extra2/block-selection-for-OOD-detection/data/wbandar1/datasets/'
    train_data = dset.SVHN(folder, split='train', transform=train_transform_cifar , download=True)
    
    test_data = dset.SVHN(folder, split='test', transform=test_transform_cifar , download=True)

    are_labels_same = np.array_equal(np.array(train_data.targets), npy_file_path_train)
    if are_labels_same:
            
        print("Labels from svhn train dataset and ground truth labels are the same.")
    else:
        # test_data.targets =npy_file_path_test
        print("Labels from avhn train dataset and ground truth labels are different.")

    train_size = int(0.95 * len(train_data))
    valid_size = len(train_data) - train_size

    # Split the dataset into training and validation sets
    train_data, valid_data = random_split(train_data, [train_size, valid_size])


    train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True, pin_memory=True, num_workers = 4) 
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size, shuffle=False, pin_memory=True, num_workers = 4) 

    test_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False, pin_memory=True, num_workers = 4)    
    return train_loader, valid_loader

#for celeba ID training
def get_celeba_t( batch_size):
    train_data = dset.CelebA(root='./data/wbandar1/datasets', split='train', \
                                            transform=train_transform_cifar, download=True)

    test_data = dset.CelebA(root='./data/wbandar1/datasets', split='test', \
                                            transform=test_transform_cifar, download=True)
    
    are_labels_same = np.array_equal(np.array(train_data.targets), npy_file_path_train)
    if are_labels_same:
            
        print("Labels from celeba train dataset and ground truth labels are the same.")
    else:
        # test_data.targets =npy_file_path_test
        print("Labels from celeba train dataset and ground truth labels are different.")

    train_size = int(0.95 * len(train_data))
    valid_size = len(train_data) - train_size

    # Split the dataset into training and validation sets
    train_data, valid_data = random_split(train_data, [train_size, valid_size])
    
     
    train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True, pin_memory=True, num_workers = 4) 
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size, shuffle=False, pin_memory=True, num_workers = 4)  
    test_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False, pin_memory=True, num_workers = 4)    
    return train_loader, valid_loader







# for svhn OOD testing
def get_svhn(folder, batch_size, transform_imagenet = False):
    test_data = dset.SVHN(folder, split='test', transform=transform_imagenet if transform_imagenet else test_transform_cifar, download=True)
    valid_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False, pin_memory=True, num_workers = 4)    
    return valid_loader


# def get_celeba(folder, batch_size, transform_imagenet = False):
#     test_data = dset.CelebA(root='./data/wbandar1/datasets', split='test', \
#                                             transform=test_transform_cifar, download=True)
#     valid_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False, pin_memory=True, num_workers = 4)    
#     return valid_loader


#for cifar OOD testing
def get_cif10(folder, batch_size, flip = 0):
    if flip== 0:
        test_data = dset.CIFAR10(folder, train=False, transform=test_transform_cifar_vflip, download=True)
    elif flip ==1:
        test_data = dset.CIFAR10(folder, train=False, transform=test_transform_cifar_hflip, download=True)
    else:
        test_data = dset.CIFAR10(folder, train=False, transform=test_transform_cifar, download=True)
    valid_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False, pin_memory=True, num_workers = 4)    
    return valid_loader


#for fmnist OOD testing
def get_fmnist( batch_size, flip = 0):
    folder = '/home/gargatik/extra2/block-selection-for-OOD-detection/data/wbandar1/datasets/'
    if flip== 0:
        test_data = dset.FashionMNIST(folder, train=False, transform=test_transform_cifar_vflip, download=True)
    elif flip ==1:
        test_data = dset.FashionMNIST(folder, train=False, transform=test_transform_cifar_hflip, download=True)
    else:
        test_data = dset.FashionMNIST(folder, train=False, transform=test_transform_cifar, download=True)
    valid_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False, pin_memory=True, num_workers = 4)    
    return valid_loader


#for celeba OOD testing
def get_celeba( batch_size, flip = 0):
    folder = '/home/gargatik/extra2/block-selection-for-OOD-detection/data/wbandar1/datasets/'
    if flip== 0:
        test_data = dset.CelebA(root='./data/wbandar1/datasets', split='test', transform=test_transform_cifar_vflip, download=True)
    elif flip ==1:
        test_data = dset.CelebA(root='./data/wbandar1/datasets', split='test', transform=test_transform_cifar_hflip, download=True)
    else:
        test_data = dset.CelebA(root='./data/wbandar1/datasets', split='test', transform=test_transform_cifar, download=True)
    valid_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False, pin_memory=True, num_workers = 4)    
    return valid_loader


#for svhn OOD testing
def get_ssvhn( batch_size, flip = 0):
    folder = '/home/gargatik/extra2/block-selection-for-OOD-detection/data/wbandar1/datasets/'
    if flip== 0:
        test_data = dset.SVHN(folder, split='test',  transform=test_transform_cifar_vflip, download=True)
    elif flip ==1:
        test_data = dset.SVHN(folder, split='test', transform=test_transform_cifar_hflip, download=True)
    else:
        test_data = dset.SVHN(folder, split='test', transform=test_transform_cifar, download=True)
    valid_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False, pin_memory=True, num_workers = 4)    
    return valid_loader


#for mnist ID training 
def get_mnist_t(batch_size):
    folder = '/home/gargatik/extra2/block-selection-for-OOD-detection/data/wbandar1/datasets/'
    
    # Load MNIST dataset
    train_data = dset.MNIST(folder, train=True, transform=train_transform_cifar, download=True)
    test_data = dset.MNIST(folder, train=False, transform=test_transform_cifar, download=True)
    
    # Load and shuffle the train targets
    train_targets = np.load("/home/gargatik/ood/mix-bt/mnist_raw_images_train.npy")
    train_indices = np.random.permutation(len(train_targets))
    train_data.targets = torch.tensor(train_targets[train_indices], dtype=torch.long)
    
    # Load and shuffle the test targets
    test_targets = np.load("/home/gargatik/ood/mix-bt/mnist_raw_images_test.npy")
    test_indices = np.random.permutation(len(test_targets))
    test_data.targets = torch.tensor(test_targets[test_indices], dtype=torch.long)
    
    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    
    return train_loader, valid_loader


#for mnist OOD testing
def get_mnista( batch_size, flip = 0):
    folder = '/home/gargatik/extra2/block-selection-for-OOD-detection/data/wbandar1/datasets/'
    if flip== 0:
        test_data = dset.MNIST(folder, train=False, transform=test_transform_cifar_vflip, download=True)
    elif flip ==1:
        test_data = dset.MNIST(folder, train=False, transform=test_transform_cifar_hflip, download=True)
    else:
        test_data = dset.MNIST(folder, train=False, transform=test_transform_cifar, download=True)
    valid_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False, pin_memory=True, num_workers = 4)    
    return valid_loader


# #for mnist OOD testing
# def get_mnist( batch_size):
#     folder = '/home/gargatik/extra2/block-selection-for-OOD-detection/data/wbandar1/datasets/'
#     test_data = dset.MNIST(folder, train=False, transform=test_transform_cifar, download=True)
#     valid_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False, pin_memory=True, num_workers = 4)    
#     return valid_loader



def get_ood(path):
    ood_data = torchvision.datasets.ImageFolder(path, test_transform_cifar)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=100, shuffle=False, pin_memory=True)
    return ood_loader    


def get_places(path):
    ood_data = torchvision.datasets.ImageFolder(path, test_transform_cifar)

    random.seed(0)
    ood_data.samples = random.sample(ood_data.samples, 10000)

    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=100, shuffle=False, pin_memory=True)
    return ood_loader

class CifarPairTransform:
    def __init__(self, train_transform=True, pair_transform=True):
        if train_transform:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ])
        self.pair_transform = pair_transform

    def __call__(self, x, label=None):
        if self.pair_transform:
            y1 = self.transform(x)
            y2 = self.transform(x)
            if label is not None:
                return y1, y2, label
            else:
                return y1, y2
        else:
            return self.transform(x), label if label is not None else None

if __name__ == '__main__':
    pass