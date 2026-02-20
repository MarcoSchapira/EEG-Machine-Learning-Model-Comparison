import os
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

#! Class Specific Accuracy Calcs
def calculatePerClass(data_dict, metric_name='Precision'):
    metric_dict = {}
    for key in data_dict.keys():
        df = data_dict[key]
        if metric_name == 'Precision':
            metric_dict[key] = precision_score(df['true'], df['pred'], average=None, zero_division=0)
        elif metric_name == 'Recall':
            metric_dict[key] = recall_score(df['true'], df['pred'], average=None, zero_division=0)
    
    df = pd.DataFrame(metric_dict)
    df = df * 100
    df = df.applymap(lambda x: round(x, 2))
    df['mean'] = df.apply('mean', axis=1).round(2) 
    df['std'] = df.apply('std', axis=1).round(2) 
    df['metrics'] = metric_name
    return df

#! Model Evaluation Metrics
def calMetrics(y_true, y_pred):
    number = max(y_true)
    mode = 'binary' if number == 1 else 'macro'
    
    accuracy = accuracy_score(y_true, y_pred)
    precison = precision_score(y_true, y_pred, average=mode, zero_division=0)
    recall = recall_score(y_true, y_pred, average=mode, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=mode, zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    return accuracy, precison, recall, f1, kappa

#! Choosing Dataset
#Mainly using C which is our dataset but left the option for BCI competition a and b datasets in case I want to compare performance
def numberClassChannel(database_type):
    if database_type == 'A': return 4, 22
    elif database_type == 'B': return 2, 3
    elif database_type == 'C': return 12, 60
    return 0, 0

#! Segmentation and Reconstruction (S&R) data augmentation
def apply_interaug(model, timg, label, num_aug, batch_size, num_classes, num_seg, num_channels):
    """Segmentation and Reconstruction (S&R) data augmentation"""
    aug_data, aug_label = [], []
    num_records = num_aug * int(batch_size / num_classes)
    seg_points = 1000 // num_seg
    
    for clsAug in range(num_classes):
        cls_idx = np.where(label == clsAug)[0]
        if len(cls_idx) == 0: continue
        
        tmp_data = timg[cls_idx]
        tmp_label = label[cls_idx]
        if model in ('MSCFormer', 'EEGEncoder'):
            tmp_aug_data = np.zeros((num_records, 1, num_channels, 1000))
        else:
            tmp_aug_data = np.zeros((num_records, num_channels, 1000))
        
        for ri in range(num_records):
            for rj in range(num_seg):
                rand_idx = np.random.randint(0, tmp_data.shape[0], num_seg)
                # Clean up the math into variables to make it easier to read
                start_pt = rj * seg_points
                end_pt = (rj + 1) * seg_points
                
                # Apply the correct dimensional slicing!
                if model in ('MSCFormer', 'EEGEncoder'):
                    # 4D slicing: (Batch, 1, Channel, Time)
                    tmp_aug_data[ri, :, :, start_pt:end_pt] = \
                        tmp_data[rand_idx[rj], :, :, start_pt:end_pt]
                else:
                    # 3D slicing: (Batch, Channel, Time)
                    tmp_aug_data[ri, :, start_pt:end_pt] = \
                        tmp_data[rand_idx[rj], :, start_pt:end_pt]
                    
        aug_data.append(tmp_aug_data)
        aug_label.append(tmp_label[:num_records])
        
    if not aug_data: 
        return torch.empty(0).cuda(), torch.empty(0).cuda()
        
    aug_data = np.concatenate(aug_data)
    aug_label = np.concatenate(aug_label)
    
    aug_shuffle = np.random.permutation(len(aug_data))
    aug_data_tensor = torch.from_numpy(aug_data[aug_shuffle]).cuda().float()
    aug_label_tensor = torch.from_numpy(aug_label[aug_shuffle]).cuda().long()
    return aug_data_tensor, aug_label_tensor

#!Dataloading
def get_source_data(model, root_dir, nSub, evaluate_mode, val_ratio, test_ratio, cols_to_keep=None):
    """Loads, filters, splits, and normalizes EEG data for a given subject."""
    pt_path = os.path.join(root_dir, f"sub_{nSub:02d}.pt")
    subj_data = torch.load(pt_path, weights_only=False)
    
    X = subj_data['data'].numpy() 
    y = subj_data['label'].numpy()

    if cols_to_keep is not None:
        X = X[:, cols_to_keep, :]
    
    if model in ('MSCFormer', 'EEGEncoder') and X.ndim == 3:
        X = np.expand_dims(X, axis=1)

    if evaluate_mode == 'LOSO':
        X_test, y_test = X, y 
        X_train_val_list, y_train_val_list = [], []
        
        for i in range(1, 26):
            if i != nSub:
                other_path = os.path.join(root_dir, f"sub_{i:02d}.pt")
                if not os.path.exists(other_path): continue
                
                other_data = torch.load(other_path, weights_only=False)
                X_other = other_data['data'].numpy()
                if cols_to_keep is not None: X_other = X_other[:, cols_to_keep, :]
                if model in ('MSCFormer', 'EEGEncoder') and X_other.ndim == 3: X_other = np.expand_dims(X_other, axis=1)
                    
                X_train_val_list.append(X_other)
                y_train_val_list.append(other_data['label'].numpy())
                
        X_combined = np.concatenate(X_train_val_list, axis=0)
        y_combined = np.concatenate(y_train_val_list, axis=0)

        X_train, X_val, y_train, y_val = train_test_split(X_combined, y_combined, test_size=val_ratio, stratify=y_combined, random_state=42)

    else:
        #Perform test split with stratify equal y meaning trial get split evenly (same ratio of rests in each split)
        #Random state 42 shuffles data in the same exact way every time so tests are repeatable
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_ratio, stratify=y, random_state=42)
        #Find new validation percentage to make it equal to original now that dataset is smaller
        relative_val_ratio = val_ratio / (1.0 - test_ratio)
        #Get final train and val split
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=relative_val_ratio, stratify=y_temp, random_state=42)

    target_mean = np.mean(X_train)
    target_std = np.std(X_train)
    
    X_train = (X_train - target_mean) / target_std
    X_val = (X_val - target_mean) / target_std
    X_test = (X_test - target_mean) / target_std
    
    return X_train, y_train, X_val, y_val, X_test, y_test




