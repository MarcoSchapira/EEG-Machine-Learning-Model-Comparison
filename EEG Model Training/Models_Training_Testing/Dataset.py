import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

class FocalLoss(nn.Module):
    """
    Focal Loss dynamically scales the loss based on prediction confidence.
    Hard examples (Movement) get full penalty. Easy examples (Rest) get reduced penalty.
    """
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing
        
    def forward(self, inputs, targets):
        # 1. Compute standard Cross Entropy Loss (supports weights and smoothing)
        ce_loss = F.cross_entropy(
            inputs, targets, weight=self.weight, 
            label_smoothing=self.label_smoothing, reduction='none'
        )
        
        # 2. Get the probability of the true target class (pt)
        log_probs = F.log_softmax(inputs, dim=-1)
        pt = torch.exp(log_probs.gather(1, targets.unsqueeze(1)).squeeze(1))
        
        # 3. Calculate the Focal Weight: (1 - pt)^gamma
        # If pt is high (confident), weight drops to 0. If pt is low (wrong), weight stays near 1.
        focal_weight = (1 - pt) ** self.gamma
        
        # 4. Apply the weight to the CE loss and average across the batch
        loss = focal_weight * ce_loss

        if self.weight is not None:
            return loss.sum() / self.weight[targets].sum()
        else:
            return loss.mean()
       
        
        #return loss.mean()


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
    elif database_type == 'C': return 9, 60
    return 0, 0

#! Segmentation and Reconstruction (S&R) data augmentation
def apply_interaug(model, timg, label, num_aug, batch_size, num_classes, num_seg, num_channels):
    """Segmentation and Reconstruction (S&R) data augmentation"""
    aug_data, aug_label = [], []
    seg_points = 1000 // num_seg
    
    for clsAug in range(num_classes):
        # Skip the Rest class entirely to prevent exacerbating the class imbalance
        if clsAug == (num_classes - 1):
            continue
            
        cls_idx = np.where(label == clsAug)[0]
        
        # We need at least 2 samples of a class in the batch to actually mix them!
        if len(cls_idx) < 2: 
            continue 
        
        tmp_data = timg[cls_idx]
        tmp_label = label[cls_idx]
        
        # Calculate records dynamically based on how many samples actually exist in this batch
        num_records_for_class = num_aug * len(cls_idx)
        
        if model in ('MSCFormer', 'EEGEncoder'):
            tmp_aug_data = np.zeros((num_records_for_class, 1, num_channels, 1000))
        else:
            tmp_aug_data = np.zeros((num_records_for_class, num_channels, 1000))
        
        for ri in range(num_records_for_class):
            for rj in range(num_seg):
                rand_idx = np.random.randint(0, tmp_data.shape[0], num_seg)
                
                start_pt = rj * seg_points
                end_pt = (rj + 1) * seg_points
                
                if model in ('MSCFormer', 'EEGEncoder'):
                    tmp_aug_data[ri, :, :, start_pt:end_pt] = \
                        tmp_data[rand_idx[rj], :, :, start_pt:end_pt]
                else:
                    tmp_aug_data[ri, :, start_pt:end_pt] = \
                        tmp_data[rand_idx[rj], :, start_pt:end_pt]
                    
        aug_data.append(tmp_aug_data)
        aug_label.append(np.full(num_records_for_class, clsAug))
        
    if not aug_data: 
        return torch.empty(0).cuda(), torch.empty(0).cuda()
        
    aug_data = np.concatenate(aug_data)
    aug_label = np.concatenate(aug_label)
    
    aug_shuffle = np.random.permutation(len(aug_data))
    aug_data_tensor = torch.from_numpy(aug_data[aug_shuffle]).cuda().float()
    aug_label_tensor = torch.from_numpy(aug_label[aug_shuffle]).cuda().long()
    
    return aug_data_tensor, aug_label_tensor

#!Dataloading
def get_source_data(model, root_dir, nSub, evaluate_mode, test_ratio, cols_to_keep, make_easier):
    """Loads, filters, splits, and normalizes EEG data for a given subject."""
    pt_path = os.path.join(root_dir, f"sub_{nSub:02d}.pt")
    subj_data = torch.load(pt_path, weights_only=False)
    
    X = subj_data['data'].numpy() 
    y_raw = subj_data['label'].numpy()

    # --- CLASS MERGING LOGIC ---
    # Merge 6,7,8 into Class 6 | Merge 9,10 into Class 7 | Shift 11 (Rest) to Class 8
    if make_easier == True:
        y = np.copy(y_raw)
        #y[np.isin(y_raw, [0, 1])] = 0
        #y[np.isin(y_raw, [2, 3])] = 1
        #y[np.isin(y_raw, [4, 5])] = 2
        y[np.isin(y_raw, [6, 7, 8])] = 6
        y[np.isin(y_raw, [9, 10])] = 7
        y[y_raw == 11] = 8

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
                y_other_raw = other_data['label'].numpy()
                
                if make_easier == True:
                    # Apply the exact same merging logic to the LOSO data
                    y_other = np.copy(y_other_raw)
                    y_other[np.isin(y_other_raw, [6, 7, 8])] = 6
                    y_other[np.isin(y_other_raw, [9, 10])] = 7
                    y_other[y_other_raw == 11] = 8
                else: y_other = y_other_raw
                
                if cols_to_keep is not None: X_other = X_other[:, cols_to_keep, :]
                if model in ('MSCFormer', 'EEGEncoder') and X_other.ndim == 3: X_other = np.expand_dims(X_other, axis=1)
                    
                X_train_val_list.append(X_other)
                y_train_val_list.append(y_other)
                
        X_train_val = np.concatenate(X_train_val_list, axis=0)
        y_train_val = np.concatenate(y_train_val_list, axis=0)

        #X_train, X_val, y_train, y_val = train_test_split(X_combined, y_combined, test_size=val_ratio, stratify=y_combined, random_state=42)

    else:
        #Perform test split with stratify equal y meaning trial get split evenly (same ratio of rests in each split)
        #Random state 42 shuffles data in the same exact way every time so tests are repeatable
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_ratio, stratify=y, random_state=42)
        #Find new validation percentage to make it equal to original now that dataset is smaller
        # = val_ratio / (1.0 - test_ratio)
        #Get final train and val split
        #X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=relative_val_ratio, stratify=y_temp, random_state=42)

    #target_mean = np.mean(X_train)
    #target_std = np.std(X_train)
    
    #X_train = (X_train - target_mean) / target_std
    #X_val = (X_val - target_mean) / target_std
    #X_test = (X_test - target_mean) / target_std
    #! Return un-normalized data. We will normalize per-fold in the main script.
    return X_train_val, y_train_val, X_test, y_test


def generate_mixup_batch(x, y, alpha=0.2):
    """
    Generates a Mixup batch by blending pairs of EEG trials.
    alpha controls how intensely the signals are mixed. 0.2 is standard.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    # Generate a random permutation of indices to pair up trials
    index = torch.randperm(batch_size).cuda()
    
    # Mix the inputs: lam * Trial A + (1 - lam) * Trial B
    mixed_x = lam * x + (1 - lam) * x[index, :]
    
    # Keep track of both original labels
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calculates the loss against both original labels and blends the result.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)