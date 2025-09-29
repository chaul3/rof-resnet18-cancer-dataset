import os
import torch
import torch.nn as nn
import sys
import numpy as np
import tqdm
import pandas as pd
import pickle
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from functools import partial
from isic import get_transform, get_loader
from isic import ISICDataset
from utils import evaluate, get_y_p

import random
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data
    target_resolution = (224, 224)
    train_transform = get_transform(target_resolution=target_resolution, train=True, augment_data=True)
    test_transform = get_transform(target_resolution=target_resolution, train=False, augment_data=False)
    trainset = ISICDataset(basedir='/Users/smileeee/Documents/Resnet18-xai/multi_groups', split="train", transform=train_transform)
    testset = ISICDataset(basedir='/Users/smileeee/Documents/Resnet18-xai/multi_groups', split="test", transform=test_transform)
    valset = ISICDataset(basedir='/Users/smileeee/Documents/Resnet18-xai/multi_groups', split="val", transform=test_transform)


    loader_kwargs = {'batch_size': 64, 'num_workers': 1, 'pin_memory': True}
    train_loader = get_loader(trainset, train=True, **loader_kwargs)
    test_loader = get_loader(testset, train=False, **loader_kwargs)
    get_yp_func = partial(get_y_p, n_places=trainset.n_places)

    # Model
    n_classes = trainset.n_classes
    model = models.resnet18(pretrained=False)
    d = model.fc.in_features
    model.fc = torch.nn.Linear(d, 2)
    checkpoint = torch.load('./resnet18_isic.pt', map_location=torch.device('cpu' if device == 'cpu' else device))
    model.load_state_dict(checkpoint)

    model = model.to(device)
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    momentum_decay = 0.9  # Set your desired momentum value
    weight_decay = 1e-4   # Set your desired weight decay value
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=momentum_decay, weight_decay=weight_decay)


    # Evaluate baseline model
    #accuracy = evaluate(model, test_loader, get_yp_func)
    #print(accuracy)
    print("Baseline model evaluation complete.")
    #
    #best_acc = 0.0  # Track the best accuracy
    #save_path = "./models/retrain_ckpt/resnet18_wb_best_baseline.pth"
    ## Training loop
    #for epoch in range(10):  # Number of epochs
    #    model.train()
    #    running_loss = 0.0
    #    for x, y, g, p, n in train_loader:
    #        x = x.to(device).requires_grad_()
    #        y = y.to(device)
    #        p = p.to(device)
    #        outputs = model(x)
    #        loss = criterion(outputs, y)
    #        
    #        optimizer.zero_grad()
    #        loss.backward()
    #        optimizer.step()
    #        running_loss += loss.item()
    #    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')
    #
    #    accuracy = evaluate(model, test_loader, get_yp_func)
    #    print(accuracy)
    #    # Save best model
    #    #if accuracy['mean_accuracy'] > best_acc:
    #    #    best_acc = accuracy['mean_accuracy']
    #    #    torch.save(model.state_dict(), save_path)
    #
    #print('Training complete') 
    # ...existing code...
    # 1. Get baseline accuracy dict
    baseline_result = evaluate(model, test_loader, get_yp_func)
    print(baseline_result)

    # Extract penultimate features from test set
    def extract_penultimate_features(model, dataloader, device):
        model.eval()
        feats, labels_all = [], []
        with torch.no_grad():
            for x, y, *_ in dataloader:
                x = x.to(device)
                # Forward up to avgpool (second-to-last layer)
                f = model.conv1(x)
                f = model.bn1(f)
                f = model.relu(f)
                f = model.maxpool(f)
                f = model.layer1(f)
                f = model.layer2(f)
                f = model.layer3(f)
                f = model.layer4(f)
                f = model.avgpool(f)
                f = torch.flatten(f, 1)
                feats.append(f.cpu())
                labels_all.append(y)
        features = torch.cat(feats, dim=0)
        labels = torch.cat(labels_all, dim=0)
        return features, labels

    features, labels = extract_penultimate_features(model, test_loader, device)
    print(f"Number of units in penultimate layer: {features.shape[1]}")
    # 2. Rank units by L1-norm
    l1_norms = features.abs().mean(dim=0)
    unit_ranking = torch.argsort(l1_norms, descending=True)

    # 2. ROF: Mask penultimate features, activate top-n units incrementally, collect all 6 accs
    def rof_penultimate_eval_multiacc(features, labels, model, model_fc, unit_ranking, top_k_list, device, test_loader, get_yp_func):
        model_fc = model_fc.to(device)
        labels = labels.to(device)
        acc_dict = {k: [] for k in baseline_result.keys() if "accuracy" in k}
        for k in top_k_list:
            mask = torch.zeros(features.shape[1])
            mask[unit_ranking[:k]] = 1.0
            masked_feats = features * mask
            masked_feats = masked_feats.to(device)

            # Patch model.forward to use masked features for this eval
            orig_forward = model.forward
            batch_start = 0
            def new_forward(x):
                nonlocal batch_start
                batch_size = x.shape[0]
                out = model_fc(masked_feats[batch_start:batch_start+batch_size])
                batch_start += batch_size
                return out
            model.forward = new_forward

            # Evaluate with current masking
            batch_start = 0
            accs = evaluate(model, test_loader, get_yp_func)
            for key in acc_dict:
                acc_dict[key].append(accs[key])

            # Restore original forward
            model.forward = orig_forward

            if k <= 10 or k % 50 == 0 or k == features.shape[1]:
                print(f"Top {k:3d} units activated | " + " | ".join([f"{key}: {accs[key]*100:.2f}%" for key in acc_dict]))
        return acc_dict

    top_k_list = list(range(1, features.shape[1]+1))
    acc_curves = rof_penultimate_eval_multiacc(
        features, labels, model, model.fc, unit_ranking, top_k_list, device, test_loader, get_yp_func
    )

    # 3. Plot all 6 accuracy curves with their baselines
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 7))
    for key in acc_curves.keys():
        plt.plot(top_k_list, acc_curves[key], label=key)
    plt.axhline(y=baseline_result['mean_accuracy'], color='r', linestyle='--', label='Baseline')
    plt.xlabel('Number of Activated Units')
    plt.ylabel('Accuracy')
    plt.title('ROF: Accuracy vs. Number of Activated Units')
    plt.legend()
    plt.show()
