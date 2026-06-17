import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import argparse

from utils import *
import resnet
import wrn
import vgg

parser = argparse.ArgumentParser()
parser.add_argument('--net','-n', default = 'resnet18', type=str)
parser.add_argument('--data', '-d', type=str)
parser.add_argument('--gpu', '-g', type=str)
parser.add_argument('--save_path', '-s', type=str)
parser.add_argument('--method' ,'-m', default = 'featurenorm', type=str)
args = parser.parse_args()

def calculate_norm(model, loader, device):
    #FeatureNorm from penultimate block
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_idx, (inputs, t) in enumerate(loader):
            x = inputs.to(device)         
            # ResNet
            features = model.forward_features_blockwise(x)
            # print("Total number of modules in self.f:", len(features))
            features = features[7]

            # Norm calculation
            norm = torch.norm(F.relu(features), dim=[2, 3])
            predictions.append(norm)
    predictions = torch.cat(predictions).to(device)
    return predictions            

def calculate_msp(model, loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_idx, (inputs, t) in enumerate(loader):
            x = inputs.to(device)         
            # ResNet
            x = model.conv1(x)
            x = model.bn1(x)
            x = model.act1(x)

            x = model.layer1[0](x)
            x = model.layer1[1](x)                   
            x = model.layer2[0](x)
            x = model.layer2[1](x)
            x = model.layer3[0](x)    
            x = model.layer3[1](x)  
            x = model.layer4[0](x)          
            x = model.layer4[1](x)      
            x = model.global_pool(x).view(-1, 512)
            x = model.fc(x)
            x = torch.softmax(x, dim=1).max(dim=1).values
            predictions.append(x)
    predictions = torch.cat(predictions).to(device)
    return predictions   

if args.method == 'msp':
    calculate_score = calculate_msp
elif args.method == 'featurenorm':
    calculate_score = calculate_norm



def get_measures(_pos, _neg, recall_level=recall_level_default):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr


def show_performance(pos, neg, method_name='Ours', recall_level=recall_level_default, file=None,rett = None):
    '''
    :param pos: 1's class, class to detect, outliers, or wrongly predicted
    example scores
    :param neg: 0's class scores
    '''

    auroc, aupr, fpr = get_measures(pos[:], neg[:], recall_level)

    print('\t\t\t' + method_name)
    print('FPR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fpr))
    print('AUROC:\t\t\t{:.2f}'.format(100 * auroc))
    print('AUPR:\t\t\t{:.2f}'.format(100 * aupr))
    # print('FDR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fdr))
    if not file is None:
        file.write('{}\n'.format(method_name))
        file.write('FPR{:d}:\t\t\t{:.2f}\n'.format(int(100 * recall_level), 100 * fpr))
        file.write('AUROC:\t\t\t{:.2f}\n'.format(100 * auroc))
        file.write('AUPR:\t\t\t{:.2f}\n'.format(100 * aupr))
        file.write('\n')
    if not rett is None:
        return fpr, auroc, aupr


def print_measures(auroc, aupr, fpr, method_name='Ours', recall_level=recall_level_default):
    print('\t\t\t\t' + method_name)
    print('  FPR{:d} AUROC AUPR'.format(int(100*recall_level)))
    print('& {:.2f} & {:.2f} & {:.2f}'.format(100*fpr, 100*auroc, 100*aupr))
    #print('FPR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fpr))
    #print('AUROC: \t\t\t{:.2f}'.format(100 * auroc))
    #print('AUPR:  \t\t\t{:.2f}'.format(100 * aupr))


# def average_ensemble(preds1, preds2):
#     return (preds1 + preds2) / 2

# def weighted_average_ensemble(preds1, preds2, weight1=0.8, weight2=0.5):
#     return (weight1 * preds1 + weight2 * preds2) / (weight1 + weight2)

# def soft_voting_ensemble(preds1, preds2):
#     print(preds1.shape,"pred shape")
#     combined_preds = torch.stack([preds1, preds2])
#     majority_vote, _ = torch.mode(combined_preds, dim=0)
#     return majority_vote # For simplicity, same as averaging

def soft_voting_ensemble(preds1, preds2,preds3):
    # print(preds1.shape,"pred shape")
    combined_preds = torch.stack([preds2,preds3, preds1])
    # print(combined_preds.shape,"combined_preds shape")
    num_channels = combined_preds.size(2)  # Get the number of channels (C)
    # print('num_channels',num_channels)
    repeat_counts = torch.zeros(combined_preds.size(1),combined_preds.size(2))  # Initialize count for each channel
    
    # Loop over each channel
    for channel_a in range(combined_preds.size(1)):

        for channel in range(combined_preds.size(2)):


            # For each channel, get the values across dim=0 (i.e., across different predictions)
            channel_vals = combined_preds[:,channel_a, channel]  # Shape: (3,) values for this channel
            # print(channel_vals.shape,"channel_vals")
            # print(channel_vals)

        
            # Check if at least two values are the same in the channel
            unique_vals = torch.unique(channel_vals)  # Get the unique values
            if unique_vals.size(0) < 3:  # If there are fewer than 3 unique values, at least two are the same
                repeat_counts[channel_a, channel] = 1  # Mark this channel as having at least two similar values
    
    # print("Repeat counts per channel:", torch.sum(repeat_counts))
    # print("Repeat counts per channel:", repeat_counts.shape)
    # majority_vote, _ = torch.mode(combined_preds, dim=0)
    # majority_vote = torch.mean(combined_preds, dim=0)#this is 
    majority_vote, _ = torch.min(combined_preds, dim=0)
    majority_vote = majority_vote.mean(1)
    # print(majority_vote.shape)
    return majority_vote # For simplicity, same as averaging

def bagging_ensemble(preds_list):
    return torch.mean(torch.stack(preds_list), dim=0)

def OOD_results(preds_id, models, loader, device, method, ensemble_method, file):
    preds_ood_list = [calculate_score(model, loader, device).cpu() for model in models]

    
    if ensemble_method == 'soft_voting':
        preds_ood = soft_voting_ensemble(preds_ood_list[0], preds_ood_list[1], preds_ood_list[2])  # Assuming only two models for soft voting
    
    else:
        raise ValueError(f"Unknown ensemble method: {ensemble_method}")

    print(torch.mean(preds_ood), torch.mean(preds_id))
    fpr, auroc, aupr = show_performance(preds_id, preds_ood, method, file=file, rett=45)
    return fpr * 100, auroc * 100, aupr * 100



    
def eval():
    config = read_conf('conf/'+args.data+'.json')
    device = 'cuda:'+args.gpu
    dataset_path = config['id_dataset']
    batch_size = config['batch_size']
    save_path = config['save_path'] + args.save_path
    # ensemble_method = 'average'  # Change this to 'weighted_average' or 'soft_voting' as needed
    # ensemble_method = 'bagging'
    ensemble_method = 'soft_voting'
    
    num_classes = int(config['num_classes'])

    if 'cifar' in args.data:
        train_loader, valid_loader = get_cifar(args.data, dataset_path, batch_size, eval=True)
    
    if args.net == 'resnet18':
        from model2 import Model2
        import model_wrn
        dataset = 'cifar10'
        from model import Model
        model2 = Model(1024, dataset, 'resnet18').to(device)
        from thop import profile, clever_format
        flops, params = profile(model2, inputs=(torch.randn(1, 3, 32, 32).to(device),))

        flops, params = clever_format([flops, params])
        import copy
        model3 = copy.deepcopy(model2)
        model2.load_state_dict((torch.load('/home/gargatik/extra2/res18_cifar10/cif10_kmeans20/model2/model_best.pth.tar', map_location = device)['state_dict']))
        model3.load_state_dict((torch.load('/home/gargatik/extra2/res18_cifar10/cif10_kmeans10/model2/model_best.pth.tar', map_location = device)['state_dict']))
        model2.to(device)
        model3.to(device)
        model2.eval()
        model3.eval()

    model_paths = [
            '/home/gargatik/extra2/res18_cifar10/cif10_kmeans20/model2/model_best.pth.tar',
            '/home/gargatik/extra2/res18_cifar10/cif10_kmeans10/model2/model_best.pth.tar',
            '/home/gargatik/extra2/res18_cifar10/cif10_kmeans5/model2/model_best.pth.tar'

        ]
    
    models = []
    for path in model_paths:
        model = Model(1024, dataset, 'resnet18').to(device)
        from thop import profile, clever_format
        flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).to(device),))

        flops, params = clever_format([flops, params])
        model.load_state_dict(torch.load(path, map_location=device)['state_dict'])
        
        model.eval()
        models.append(model)

    f = open(save_path+'/{}_result.txt'.format(args.method), 'w')
    # preds_in = calculate_score(models[1], valid_loader, device).cpu()  # Use the first model for ID predictions
    preds_ood_list = [calculate_score(model, valid_loader, device).cpu() for model in models]  # Use the first model for ID predictions
    preds_in = soft_voting_ensemble(preds_ood_list[0], preds_ood_list[1], preds_ood_list[2])
    
    fpr1, auroc1, aupr1 = OOD_results(preds_in, models, get_ood('../../ood/block-selection-for-OOD-detection/data/ood_test/dtd/images'), device, args.method + '-TEXTURES', ensemble_method, f)
    fpr2, auroc2, aupr2 = OOD_results(preds_in, models, get_svhn('../../ood/block-selection-for-OOD-detection/data/ood_test/svhn', batch_size), device, args.method + '-SVHN', ensemble_method, f)
    fpr3, auroc3, aupr3 = OOD_results(preds_in, models, get_ood('../../ood/block-selection-for-OOD-detection/data/ood_test/LSUN'), device, args.method + '-LSUN-crop', ensemble_method, f)
    fpr4, auroc4, aupr4 = OOD_results(preds_in, models, get_ood('../../ood/block-selection-for-OOD-detection/data/ood_test/LSUN_pil'), device, args.method + '-LSUN-resize', ensemble_method, f)
    fpr5, auroc5, aupr5 = OOD_results(preds_in, models, get_ood('../../ood/block-selection-for-OOD-detection/data/ood_test/iSUN'), device, args.method + '-iSUN', ensemble_method, f)
    fpr6, auroc6, aupr6 = OOD_results(preds_in, models, get_places('../../ood/block-selection-for-OOD-detection/data/ood_test/places_256'), device, args.method + '-Places365', ensemble_method, f)

    import pandas as pd
    avg_auroc = (auroc1 + auroc2 + auroc3 + auroc4 + auroc5 + auroc6) / 6
    avg_fpr = (fpr1 + fpr2 + fpr3 + fpr4 + fpr5 + fpr6) / 6
    avg_aupr = (aupr1 + aupr2 + aupr3 + aupr4 + aupr5 + aupr6) / 6
    print(avg_auroc)
    print(avg_fpr)

    # Create a DataFrame with the calculated values
    data = {
        'Datasets': ['Textures', 'SVHN', 'LSUN-C', 'LSUN-R', 'iSUN', 'Places365'],
        'AUROC': [auroc1, auroc2, auroc3, auroc4, auroc5, auroc6],
        'FPR': [fpr1, fpr2, fpr3, fpr4, fpr5, fpr6]
    }

    df = pd.DataFrame(data)
    df = df.set_index('Datasets').T
    output_file = 'performance_results1.xlsx'
    df.to_excel(output_file)
    f.close()

if __name__ == '__main__':
    eval()
