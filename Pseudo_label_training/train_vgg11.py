# Acknowledgement
# This method was build upon codes of https://github.com/gist-ailab/block-selection-for-OOD-detection/tree/main
# & https://github.com/wgcban/mix-bt







import os, copy
import torch
import torchvision
from torchvision import transforms
import argparse
import numpy as np
import timm
import torch.nn.functional as F
import utils
import resnet
import wrn
import vgg
import wandb
  



def list_of_strings(arg):
    return arg.split(',')
def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def calculate_norm(model, loader, device):
    #FeatureNorm from penultimate block
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_idx, (inputs, t) in enumerate(loader):
            x = inputs.to(device)         
            # ResNet
            features = model.forward_features_blockwise(x)

            features = features[5]

            # Norm calculation
            norm = torch.norm(F.relu(features), dim=[2, 3]).mean(1)
            predictions.append(norm)
    predictions = torch.cat(predictions).to(device)
    return predictions

def OOD_results(preds_id, model, loader, device, method, file):  
    #image_norm(loader)
    preds_ood = calculate_norm(model, loader, device).cpu()

    print(torch.mean(preds_ood), torch.mean(preds_id))
    fpr, auroc, aupr = show_performance(preds_id, preds_ood, method, file=file,rett= 45)
    return fpr*100, auroc*100, aupr*100


from utils import *




def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net','-n', default = 'resnet18', type=str)
    parser.add_argument('--data', '-d', type=str)
    parser.add_argument('--gpu', '-g', default = '2', type=str)
    parser.add_argument('--save_path', '-s', type=str)
    parser.add_argument('--seed', type=int, default = 0)

    
    
    args = parser.parse_args()
    config = utils.read_conf('conf/'+args.data+'.json')
    device = 'cuda:'+args.gpu
    torch.manual_seed(args.seed) 
    model_name = args.net
    dataset_path = config['id_dataset']
    save_path = config['save_path'] + args.save_path
    num_classes = int(config['num_classes'])
    class_range = list(range(0, num_classes))
    print(args.gpu,"args.gpu")


    wandb.init(project = 'block_res18new____',name="res18_orig_o2")
    
    batch_size = int(config['batch_size'])
    max_epoch = int(config['epoch'])
    wd = 5e-04
    lrde = [50, 75, 90]
    lr = 0.1
    num_blocks = 8
    

    print(model_name, dataset_path.split('/')[-2], batch_size, class_range)
    
    if not os.path.exists(config['save_path']):
        os.mkdir(config['save_path'])
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    else:
        pass
        # raise ValueError('save_path already exists')
    
    block_info_path = save_path + '/block_info.txt'
    origin_loader, jigsaw_loader = utils.get_cifar_jigsaw(args.data, dataset_path, batch_size)

    if 'sudo' in args.data and 'cifar' in args.data:
        print(args.data)
        train_loader, valid_loader = utils.get_sudo_cifar(args.data, dataset_path, batch_size, eval = False, config = config)

    
    elif 'cifar' in args.data:
        train_loader, valid_loader = utils.get_cifar(args.data, dataset_path, batch_size)


    from model2 import Model2
    dataset = 'cifar10' 
    



    import model_vgg
    model2 = model_vgg.VGG(vgg_name = 'VGG11', num_classes = 1024).to(device)
    model2.load_state_dict(torch.load("vgg ssl checkpoint address", map_location='cpu'), strict=True)


    

    
    model = Model2(1024,dataset , 'vgg11',num_classes).to(device)
    model.to(device)
    model2.to(device)

    
    criterion = torch.nn.CrossEntropyLoss()    
    optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum=0.9, weight_decay = wd)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lrde)

    optimizer2 = torch.optim.SGD(model2.parameters(), lr = lr, momentum=0.9, weight_decay = wd)

    scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer2, lrde)
    save_path2 = os.path.join(save_path, "model2")
    os.mkdir(save_path2)
    saver = timm.utils.CheckpointSaver(model, optimizer, checkpoint_dir= save_path, max_history = 10,checkpoint_prefix = 'saver') 
    saver2 = timm.utils.CheckpointSaver(model2, optimizer2, checkpoint_dir= save_path2, max_history = 10,checkpoint_prefix = 'saver2')      
    args.method ='featurenorm'
    max_ratio = 0
    mixup_loss_scale= 5
    total_loss_bt = 0
    total_loss_mix = 0
    total_loss_c = 0
    for epoch in range(max_epoch):
        
        ## training
        model.train()
        model2.train()
        total_loss = 0
        total = 0
        correct = 0
        total_loss_bt = 0
        total_loss_mix = 0
        total_loss_c = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)


            







            optimizer.zero_grad()
            optimizer2.zero_grad()


            orig, out = model2(inputs)
            outputs = model(orig)
                


            loss = criterion(outputs, targets.long())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping 
            torch.nn.utils.clip_grad_norm_(model2.parameters(), max_norm=1.0)  # Gradient clipping           
            optimizer.step()

            optimizer2.step()

            total_loss += loss
            total += targets.size(0)
            _, predicted = outputs.max(1)            
            correct += predicted.eq(targets).sum().item()            
            print('\r', batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (total_loss/(batch_idx+1), 100.*correct/total, correct, total), end = '')                       
        train_accuracy = correct/total
        train_avg_loss = total_loss/len(train_loader)
        # Calculate average losses
        avg_loss_bt = total_loss_bt / len(train_loader)
        avg_loss_mix = total_loss_mix / len(train_loader)
        avg_loss_c = total_loss_c / len(train_loader)
        print()

        ## validation
        model.eval()
        model2.eval()
        total_loss = 0
        total = 0
        correct = 0
        valid_accuracy = utils.validation_accuracy2(model, model2, valid_loader, device)

        ratio =0

        scheduler.step()
        scheduler2.step()
 

        if epoch%1 == 0:
            f = open(save_path+'/{}_result.txt'.format(args.method), 'w')
            valid_accuracy = validation_accuracy2(model, model2, valid_loader, device)
            print(valid_accuracy)
            f.write('Accuracy for ValidationSet: {}\n'.format(str(valid_accuracy)))
            preds_in = calculate_norm(model2, valid_loader, device).cpu()
            fpr1, auroc1, aupr1 = OOD_results(preds_in, model2, get_ood('../../ood/block-selection-for-OOD-detection/data/ood_test/dtd/images'), device, args.method+'-TEXTURES', f) # Textures
            fpr2, auroc2, aupr2 =OOD_results(preds_in, model2, get_svhn('../../ood/block-selection-for-OOD-detection/data/ood_test/svhn', batch_size), device, args.method+'-SVHN', f)
            
            fpr3, auroc3, aupr3 =OOD_results(preds_in, model2, get_ood('../../ood/block-selection-for-OOD-detection/data/ood_test/LSUN'), device, args.method+'-LSUN-crop', f) # LSUN(c)
            fpr4, auroc4, aupr4 =OOD_results(preds_in, model2, get_ood('../../ood/block-selection-for-OOD-detection/data/ood_test/LSUN_pil'), device, args.method+'-LSUN-resize', f) #LSUN(r)
            fpr5, auroc5, aupr5 =OOD_results(preds_in, model2, get_ood('../../ood/block-selection-for-OOD-detection/data/ood_test/iSUN'), device, args.method+'-iSUN', f) #iSUN

            fpr6, auroc6, aupr6 =OOD_results(preds_in, model2, get_places('../../ood/block-selection-for-OOD-detection/data/ood_test/places_256'), device, args.method+'-Places365', f)
            import pandas as pd

            # Assuming you have calculated all the values as mentioned in your code snippet
            avg_auroc = (auroc1 + auroc2 + auroc3 + auroc4 + auroc5 + auroc6) / 6
            avg_fpr = (fpr1 + fpr2 + fpr3 + fpr4 + fpr5 + fpr6) / 6
            avg_aupr = (aupr1 + aupr2 + aupr3 + aupr4 + aupr5 + aupr6) / 6


            # Log averages using WandB
            wandb.log({"Average AUROC": avg_auroc})
            wandb.log({"Average FPR": avg_fpr})
            wandb.log({"valid_acc": valid_accuracy })

            
            # Create a DataFrame with the calculated values
            data = {
                'Datasets': ['Textures', 'SVHN', 'LSUN-C', 'LSUN-R', 'iSUN', 'Places365'],
                'AUROC': [auroc1, auroc2, auroc3, auroc4, auroc5, auroc6],
                'FPR': [fpr1, fpr2, fpr3, fpr4, fpr5, fpr6]
            }

            df = pd.DataFrame(data)

            # Transpose the DataFrame to have Datasets as columns
            df = df.set_index('Datasets').T

            # Write the DataFrame to an Excel file
            output_file = 'performance_results1.xlsx'
            df.to_excel(output_file)
            f.close()
            saver.save_checkpoint(epoch, metric=avg_auroc)
            saver2.save_checkpoint(epoch, metric=avg_auroc)
        
  
            





        print('EPOCH {:4}, TRAIN [loss - {:.4f}, acc - {:.4f}], VALID [acc - {:.4f}, RATIO [ratio - {:.4f}]\n'.format(epoch, train_avg_loss, train_accuracy, valid_accuracy, ratio))


        wandb.log({"Train loss": train_avg_loss})
        wandb.log({"Train acc": train_accuracy})
        wandb.log({"loss_bt": avg_loss_bt})
        wandb.log({"loss_mix": avg_loss_mix})
        wandb.log({"loss_CE": avg_loss_c})
        wandb.log({"EPOCH": epoch})

if __name__ =='__main__':
    train()