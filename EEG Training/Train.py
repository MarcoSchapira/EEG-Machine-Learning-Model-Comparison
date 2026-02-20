import os
import numpy as np
import pandas as pd
import random
import datetime
import torch
from torch.backends import cudnn
import warnings
from MSCFormerModel import Parameters, MSCFormer
from EEGEncoderTrain import EEGEncoder
from TCNet_Model import TCNetModel
from Dataset import calMetrics, numberClassChannel, apply_interaug, get_source_data

warnings.filterwarnings("ignore")
cudnn.benchmark = False
cudnn.deterministic = True

def train_subject(model, nSub, X_train, y_train, X_val, y_val, X_test, y_test, config):
    """Handles the training loop for a single subject."""
    
    # Setup DataLoaders
    Tensor, LongTensor = torch.cuda.FloatTensor, torch.cuda.LongTensor
    train_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_train).type(Tensor), torch.from_numpy(y_train).type(LongTensor))
    val_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_val).type(Tensor), torch.from_numpy(y_val).type(LongTensor))
    test_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_test).type(Tensor), torch.from_numpy(y_test).type(LongTensor))
    
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False)

    # 1. Dynamically set weight decay
    # EEGEncoder handles its own L2 regularization, so we turn it off here to prevent double-penalization
    if config['model_name'] == 'EEGEncoder':
        current_weight_decay = 0.0
    else:
        current_weight_decay = config.get('weight_decay', 1e-4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], betas=(0.5, 0.999), weight_decay=current_weight_decay)
    class_weights = torch.ones(config['num_classes']).to('cuda') * 12.0
    class_weights[11] = 1.0 
    loss_func = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.2)
    
    best_epoch = 0
    min_loss = float('inf')
    result_process = []
    model_filename = os.path.join(config['result_dir'], f'model_{nSub}.pth')
    
    print(f"\n--- Starting Training for Subject {nSub} ---")
    
    # --- EPOCH LOOP ---
    for e in range(config['epochs']):
        model.train()
        running_train_loss = 0.0
        train_outputs, train_labels = [], []

        for img, label in train_dl:
            # Data Augmentation
            aug_img, aug_lbl = apply_interaug(
                config['model_name'], X_train, y_train, config['n_aug'], config['batch_size'], 
                config['num_classes'], config['n_seg'], config['num_channels']
            )
            if len(aug_img) > 0:
                img = torch.cat((img, aug_img))
                label = torch.cat((label, aug_lbl))

            # Shuffle combined batch
            shuffle_idx = torch.randperm(img.size(0))
            img, label = img[shuffle_idx], label[shuffle_idx]

            # Forward & Backprop
            features, outputs = model(img)
            loss = loss_func(outputs, label) 
            
            l2_reg = sum(m.l2_loss() for m in model.modules() if hasattr(m, 'l2_loss'))
            total_loss = loss + l2_reg
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            train_outputs.append(outputs.detach())
            train_labels.append(label.detach())
            running_train_loss += loss.item()

        # Epoch Train Metrics
        all_t_out = torch.cat(train_outputs)
        all_t_lbl = torch.cat(train_labels)
        train_acc = float((torch.max(all_t_out, 1)[1] == all_t_lbl).cpu().numpy().astype(int).sum()) / float(all_t_lbl.size(0))
        
        # --- VALIDATION LOOP ---
        model.eval()
        running_val_loss = 0.0
        val_outputs, val_labels = [], []
        
        with torch.no_grad():
            for v_img, v_lbl in val_dl:
                _, v_out = model(v_img)
                running_val_loss += loss_func(v_out, v_lbl).item()
                val_outputs.append(v_out)
                val_labels.append(v_lbl)
                
        all_v_out = torch.cat(val_outputs)
        all_v_lbl = torch.cat(val_labels)
        val_acc = float((torch.max(all_v_out, 1)[1] == all_v_lbl).cpu().numpy().astype(int).sum()) / float(all_v_lbl.size(0))
        
        val_loss = running_val_loss / len(val_dl)
        
        result_process.append({
            'epoch': e, 'train_acc': train_acc, 'train_loss': running_train_loss / len(train_dl),
            'val_acc': val_acc, 'val_loss': val_loss
        })

        # Save Best Model
        if min_loss > val_loss:
            min_loss = val_loss
            best_epoch = e
            torch.save(model, model_filename)
            print(f"[{nSub}_{e}] Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f} (Saved)")

    # --- FINAL TEST PIPELINE ---
    model = torch.load(model_filename, weights_only=False).cuda()
    model.eval()
    test_outputs, test_labels = [], []
    
    with torch.no_grad():
        for t_img, t_lbl in test_dl:
            _, t_out = model(t_img)
            test_outputs.append(t_out)
            test_labels.append(t_lbl)
            
    all_test_outputs = torch.cat(test_outputs)
    y_true = torch.cat(test_labels)
    y_pred = torch.max(all_test_outputs, 1)[1]
    
    test_acc = float((y_pred == y_true).cpu().numpy().astype(int).sum()) / float(y_true.size(0))
    print(f"\nSubject {nSub} Final Test Accuracy: {test_acc:.4f}")

    return test_acc, y_true, y_pred, pd.DataFrame(result_process)

if __name__ == "__main__":
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    # --- CONFIGURATION DICTIONARY ---
    CONFIG = {
        'data_dir': r'D:/EEG Data/EEG_11/EEG_Compact/Processed_PerSubject_PT/',
        'model_name': 'EEGEncoder', 
        'eval_mode': 'LOSO-No',
        'dropout': 0.5,
        'dataset_type': 'C',
        
        # Split: Train ratio will be leftover percentage right now its set 70%/15%/15%
        'test_ratio': 0.15,
        'val_ratio': 0.15,  

        'cols_to_keep': list(range(4, 28)) + list(range(38, 58)),
        'n_aug': 3,
        'n_seg': 8,
        'epochs': 1000,
        'batch_size': 72,
        'lr': 0.001,
        'heads': 2,
        'depth': 6,
        'f1': 8,
        'kernel_size': 64  
    }
    
    if CONFIG['eval_mode'] == 'LOSO': CONFIG['dropout'] = 0.25

    N_SUBJECT = 25 if CONFIG['dataset_type'] == 'C' else 9
    CONFIG['num_classes'], CONFIG['num_channels'] = numberClassChannel(CONFIG['dataset_type'])

    if CONFIG['cols_to_keep']:
        CONFIG['num_channels'] = len(CONFIG['cols_to_keep'])

    CONFIG['result_dir'] = f"{CONFIG['dataset_type']}_{CONFIG['model_name']}"
    os.makedirs(CONFIG['result_dir'], exist_ok=True)

    file_process = os.path.join(CONFIG['result_dir'], "process_train.xlsx")
    file_pred = os.path.join(CONFIG['result_dir'], "pred_true.xlsx")
    subjects_result = []

    #! --- MAIN SUBJECT LOOP ---
    for sub_idx in range(1, N_SUBJECT + 1):
        # Set seeds for reproducibility
        seed_n = np.random.randint(2024)
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)

        # 1. Load Model
        if CONFIG['model_name'] == 'MSCFormer':
            model = MSCFormer(Parameters(CONFIG['dropout']), CONFIG['num_classes'], CONFIG['num_channels']).cuda() 
        elif CONFIG['model_name'] == 'TCNet':
            model = TCNetModel(CONFIG['num_classes'], CONFIG['num_channels'], 1000, CONFIG['f1'], CONFIG['kernel_size'], CONFIG['dropout']).cuda()
        #elif CONFIG['model_name'] == 'EEGNet':
            #model = EEGNetModel(CONFIG['num_classes'], CONFIG['num_channels'], 1000, CONFIG['f1'], CONFIG['kernel_size'], CONFIG['dropout']).cuda()
        elif CONFIG['model_name'] == 'EEGEncoder':
            model = EEGEncoder(CONFIG['num_classes'], CONFIG['num_channels'], 1000).cuda()

        # 2. Get Data
        X_tr, y_tr, X_v, y_v, X_te, y_te = get_source_data(
            CONFIG['model_name'], CONFIG['data_dir'], sub_idx, CONFIG['eval_mode'], 
            CONFIG['val_ratio'], CONFIG['test_ratio'], CONFIG['cols_to_keep']
        )
        
        # 3. Train
        testAcc, Y_true, Y_pred, df_process = train_subject(
            model, sub_idx, X_tr, y_tr, X_v, y_v, X_te, y_te, CONFIG
        )

        # 4. Save Excel Results
        true_cpu = Y_true.cpu().numpy().astype(int)
        pred_cpu = Y_pred.cpu().numpy().astype(int)
        df_pred_true = pd.DataFrame({'pred': pred_cpu, 'true': true_cpu})
        
        mode_p = 'a' if os.path.exists(file_process) else 'w'
        with pd.ExcelWriter(file_process, engine='openpyxl', mode=mode_p, if_sheet_exists='replace' if mode_p=='a' else None) as writer:
            df_process.to_excel(writer, sheet_name=str(sub_idx))
            
        mode_t = 'a' if os.path.exists(file_pred) else 'w'
        with pd.ExcelWriter(file_pred, engine='openpyxl', mode=mode_t, if_sheet_exists='replace' if mode_t=='a' else None) as writer:
            df_pred_true.to_excel(writer, sheet_name=str(sub_idx))

        # 5. Metrics
        accuracy, precison, recall, f1, kappa = calMetrics(true_cpu, pred_cpu)
        subjects_result.append({
            'accuracy': accuracy*100, 'precision': precison*100, 
            'recall': recall*100, 'f1': f1*100, 'kappa': kappa*100
        })

    # Final Output Summaries
    df_result = pd.DataFrame(subjects_result)
    mean, std = df_result.mean(axis=0), df_result.std(axis=0)
    mean.name, std.name = 'mean', 'std'
    df_result = pd.concat([df_result, pd.DataFrame(mean).T, pd.DataFrame(std).T])
    
    print("\n" + "-"*15 + " FINAL RESULTS " + "-"*15)
    print(df_result)