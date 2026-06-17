import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50, resnet18, resnet34
from torchvision.models.densenet import densenet121


class Model2(nn.Module):
    def __init__(self, feature_dim=128, dataset='cifar10', arch='resnet50',num_classes=128):
        super(Model2, self).__init__()

        self.f = []
        if arch == 'resnet18':
            temp_model = resnet18().named_children()
            embedding_size = 512
        elif arch == 'resnet34':
            temp_model = resnet34().named_children()
            embedding_size = 512
        elif arch == 'densenet121':
            temp_model = densenet121().features.named_children()
            embedding_size = 4096
        elif arch == 'resnet50':
            temp_model = resnet50().named_children()
            embedding_size = 2048
        elif arch == 'wrn28':
            temp_model = resnet50().named_children()
            embedding_size = 640  # Adjust the embedding size for WRN-28
        elif arch == 'wrn40':
            temp_model = resnet50().named_children()
            embedding_size = 128  # Adjust the embedding size for WRN-28
        elif arch == 'vgg11':
            temp_model = resnet50().named_children()
            embedding_size = 512  # Adjust the embedding size for WRN-28
        
        else:
            raise NotImplementedError
        
        for name, module in temp_model:
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if dataset == 'cifar10' or dataset == 'cifar100':
                if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                    self.f.append(module)
            elif dataset == 'tiny_imagenet' or dataset == 'stl10':
                if not isinstance(module, nn.Linear):
                    self.f.append(module)
        # encoder
        
        self.final = nn.Sequential(nn.Linear(embedding_size, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, num_classes, bias=True))
        self.dropout = nn.Dropout(p=0.8)
        # self.final = nn.Sequential(                             
        #     nn.Linear(512, 512),
        #     nn.ReLU(True),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(512 , 512),
        #     nn.ReLU(True),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(512, num_classes)
        # )

    def forward(self, x):
        # print(x.shape,"@@@@@@@@@@@@@@@@@")
        

        out2 = self.final(x)

        # return F.normalize(out2, dim=-1)
        return self.dropout(F.sigmoid(out2))
        # return out2
