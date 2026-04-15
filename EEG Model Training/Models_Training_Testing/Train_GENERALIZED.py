import os
import numpy as np
import pandas as pd
import random
import datetime
import torch
from torch.backends import cudnn
import warnings
from MSCFormerModel import Parameters, MSCFormer
#from EEGEncoderTrain import EEGEncoder
from EEGEncoderModel import EEGEncoder
from TCNet_Model import TCNetModel
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from Dataset import calMetrics, get_source_data, apply_interaug, generate_mixup_batch, mixup_criterion, FocalLoss

warnings.filterwarnings("ignore")
cudnn.benchmark = False
cudnn.deterministic = True

def save_to_excel(file_path, df, sheet_name):
    """Helper function to cleanly append sheets to an Excel file."""
    mode = 'a' if os.path.exists(file_path) else 'w'
    if_sheet_exists = 'replace' if mode == 'a' else None
    with pd.ExcelWriter(file_path, engine='openpyxl', mode=mode, if_sheet_exists=if_sheet_exists) as writer:
        df.to_excel(writer, sheet_name=sheet_name)

def train_subject(model, nSub, X_train, y_train, X_val, y_val, X_test, y_test, config, run_name):
    """Handles both Fold-CV and Final Production training loops."""
    Tensor, LongTensor = torch.cuda.FloatTensor, torch.cuda.LongTensor
    
    # Train & Test Dataloaders
    train_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_train).type(Tensor), torch.from_numpy(y_train).type(LongTensor))
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    test_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_test).type(Tensor), torch.from_numpy(y_test).type(LongTensor))
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False)

    # Check if we are in Validation Mode (CV Folds) or Production Mode (Final Run)
    is_cv_fold = X_val is not None and y_val is not None
    if is_cv_fold:
        val_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_val).type(Tensor), torch.from_numpy(y_val).type(LongTensor))
        val_dl = torch.utils.data.DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False)

    current_weight_decay = 0.0 if config['model_name'] == 'EEGNet' else config.get('weight_decay', 1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], betas=(0.5, 0.999), weight_decay=current_weight_decay)
    
    # --- 1. Linear Warmup Scheduler ---
    # Starts the LR at 1% of the target (0.00001) and ramps up to 100% (0.001) over 'warmup_epochs'
    warmup_epochs = config.get('warmup_epochs', 15)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_epochs
    )
    
    # --- 2. Cosine Annealing Scheduler ---
    # Takes over after the warmup and curves the LR down to 1e-6 over the remaining T_max epochs
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=(config['T_max'] - warmup_epochs), eta_min=1e-6
    )
    
    # --- 3. Sequential Scheduler ---
    # Chains them together. Switches from warmup to cosine precisely at the 'warmup_epochs' milestone.
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
    )
    
    #class_weights = torch.ones(config['num_classes']).to('cuda') * 12.0
    #class_weights[11] = 1.0 
    class_weights_np = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float).cuda()
    #loss_func = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.2)
    # Use Focal Loss with gamma=2.0 to heavily penalize misclassified minority classes
    # Label smoothing reduced to 0.05 to prevent over-softening alongside Mixup
    loss_func = FocalLoss(weight=class_weights, gamma=3.0, label_smoothing=0.05)
    
    best_epoch = 0
    min_loss = float('inf')
    patience_counter = 0
    result_process = []
    model_filename = os.path.join(config['result_dir'], f'model_{nSub}_{run_name}.pth')
    
    print(f"\n--- Starting Training for Subject {nSub} | {run_name} ---")
    
    for e in range(config['epochs']):
        # --- TRAIN LOOP ---
        model.train()
        running_train_loss = 0.0
        train_outputs, train_labels = [], []

        for img, label in train_dl:
            # Move to GPU
            img = img.cuda()
            label = label.cuda()
            
            if config['aug_mode'] in ['interaug', 'both']:
                # Data Augmentation
                aug_img, aug_lbl = apply_interaug(
                    config['model_name'], img.cpu().numpy(), label.cpu().numpy(), 
                    config['n_aug'], img.size(0), config['num_classes'], 
                    config['n_seg'], config['num_channels']
                )
                if len(aug_img) > 0:
                    img = torch.cat((img, aug_img))
                    label = torch.cat((label, aug_lbl))

                shuffle_idx = torch.randperm(img.size(0))
                img, label = img[shuffle_idx], label[shuffle_idx] 
            
            if config['aug_mode'] in ['mixup', 'both']:
                # --- 1. GENERATE MIXUP DATA ---
                # Creates an augmented batch of equal size to the original batch
                mixed_img, y_a, y_b, lam = generate_mixup_batch(img, label, alpha=0.2)

                # Concatenate original batch and Mixup batch to double the training data
                combined_img = torch.cat((img, mixed_img))

                # --- 2. FORWARD PASS ---
                features, outputs = model(combined_img)
                
                # Split outputs back into original and mixed predictions
                batch_size = img.size(0)
                outputs_orig = outputs[:batch_size]
                outputs_mixed = outputs[batch_size:]
                
                # --- 3. CALCULATE LOSS ---
                loss_orig = loss_func(outputs_orig, label)
                loss_mixed = mixup_criterion(loss_func, outputs_mixed, y_a, y_b, lam)
                loss = (loss_orig + loss_mixed) / 2.0
                
                # Track original outputs for accuracy metrics
                outputs = outputs_orig
            else:
                features, outputs = model(img)
                loss = loss_func(outputs, label)

            #l2_reg = sum(m.l2_loss() for m in model.modules() if hasattr(m, 'l2_loss'))
            #total_loss = loss + l2_reg
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_outputs.append(outputs.detach())
            train_labels.append(label.detach())
            running_train_loss += loss.item()

        all_t_out = torch.cat(train_outputs)
        all_t_lbl = torch.cat(train_labels)
        train_acc = float((torch.max(all_t_out, 1)[1] == all_t_lbl).cpu().numpy().astype(int).sum()) / float(all_t_lbl.size(0))
        
        # --- VALIDATION OR PRODUCTION SAVING ---
        if is_cv_fold:
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
            
            result_process.append({'epoch': e, 'train_acc': train_acc, 'train_loss': running_train_loss / len(train_dl), 'val_acc': val_acc, 'val_loss': val_loss})

            # Early Stopping Logic
            if min_loss > val_loss:
                min_loss = val_loss
                best_epoch = e
                patience_counter = 0 # Reset patience because we found a new best
                torch.save(model, model_filename)
                print(f"[{nSub}_{e}] Train Acc: {train_acc:.4f} | Train Loss: {(running_train_loss / len(train_dl)):.4f} | Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f} (Saved)")
            else:
                patience_counter += 1
                print(f"[{nSub}_{e}] Train Acc: {train_acc:.4f} | Train Loss: {(running_train_loss / len(train_dl)):.4f} | Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f} (Patience: {patience_counter}/{config['patience']})")
                
            if patience_counter >= config['patience']:
                print(f"--- Early stopping triggered at epoch {e}. Reverting to epoch {best_epoch} ---")
                break  # Exit the epoch loop early   
        else:
            # Production Mode: No validation set. Save model at the final epoch.
            result_process.append({'epoch': e, 'train_acc': train_acc, 'train_loss': running_train_loss / len(train_dl)})
            best_epoch = config['epochs'] - 1
            # Production Mode: No validation set, no early stopping. Train to the optimal epoch limit.
            result_process.append({'epoch': e, 'train_acc': train_acc, 'train_loss': running_train_loss / len(train_dl)})
            best_epoch = config['epochs'] - 1
            if e == best_epoch:
                torch.save(model, model_filename)
                print(f"[{nSub}_{e}] Production Train Acc: {train_acc:.4f} | Train Loss: {running_train_loss / len(train_dl):.4f} (Saved Final)")
            elif e % 10 == 0:
                print(f"[{nSub}_{e}] Production Train Acc: {train_acc:.4f} | Train Loss: {running_train_loss / len(train_dl):.4f}")

        # Step the Cosine Annealing scheduler at the end of the epoch
        scheduler.step()

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
    print(f"\n{run_name} Final Test Accuracy: {test_acc:.4f}")

    return test_acc, y_true, y_pred, pd.DataFrame(result_process), best_epoch

if __name__ == "__main__":
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    # --- CONFIGURATION DICTIONARY ---
    CONFIG = {
        'data_dir': r'/content/data/',#'/Users/marcoschapira/Documents/queens/capstone/local_data/Processed_PerSubject_PT',
        'model_name': 'MSCFormer',
        'eval_mode': 'LOSO',
        'dropout': 0.25,
        'dataset_type': 'C',
        'n_splits': 5,
        'aug_mode': 'mixup',
        # Split: Train ratio will be leftover percentage right now its set 80%/10%/10%
        'test_ratio': 0.10,  
        'T_max': 300,
        'cols_to_keep': [0, 4, 6, 8, 10, 12, 13, 15, 19, 21, 23, 25, 27, 31, 35, 37, 38, 40, 43, 45, 46, 48, 51, 53, 56, 57, 58], #list(range(4, 28)) + list(range(38, 58)), #list(range(1, 32)) + list(range(36, 65)), # None, #
        'n_aug': 2,
        'n_seg': 8,
        'epochs': 100,
        'warmup_epochs': 15,
        'patience': 5,  # Wait 30 epochs for validation loss to improve before stopping
        'batch_size': 72,
        'lr': 0.001,
        'heads': 2,
        'depth': 6,
        'f1': 8,
        'kernel_size': 64,
        'num_channels': 60,
        'num_classes': 12,
        'make_easier': True
    }
    
    if CONFIG['eval_mode'] == 'LOSO':
        CONFIG['dropout'] = 0.2

    if CONFIG['make_easier'] == True:
        CONFIG['num_classes'] = 9
    
    if CONFIG['cols_to_keep']:
        CONFIG['num_channels'] = len(CONFIG['cols_to_keep'])

    CONFIG['result_dir'] = f"{CONFIG['dataset_type']}_{CONFIG['model_name']}"
    os.makedirs(CONFIG['result_dir'], exist_ok=True)

    file_process = os.path.join(CONFIG['result_dir'], "process_train.xlsx")
    file_pred = os.path.join(CONFIG['result_dir'], "pred_true.xlsx")
    production_results = []

    #! Choose subject
    sub_idx = 1
    # Set seeds for reproducibility
    seed_n = np.random.randint(2024)
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed_all(seed_n)
    
    # Get Raw Data (No Normalization Yet)
    X_tr_raw, y_tr_raw, X_v_raw, y_v_raw, X_te_raw, y_te = get_source_data(
        CONFIG['model_name'], CONFIG['data_dir'], sub_idx, CONFIG['eval_mode'], 
        CONFIG['test_ratio'], CONFIG['cols_to_keep'], CONFIG['make_easier']
    )

    # Initialize Stratified K-Fold
    skf = StratifiedKFold(n_splits=CONFIG['n_splits'], shuffle=True, random_state=seed_n)
    fold_best_epochs = []
    fold_metrics = []

    #! --- PHASE 1: 5-Fold ---
    
    #!! skipp this phase 1 and go directly to phase 2 to save compute units

    #! Because of this, phase 2 now needs both test and validation data
    
    #! --- PHASE 2: PRODUCTION MODEL ---
    print(f"\n========== Phase 2: Production Run for Subject {sub_idx} ==========")
    
    # Normalize using TRAIN only, then apply to val and test
    target_mean, target_std = np.mean(X_tr_raw), np.std(X_tr_raw)
    X_tr_scaled = (X_tr_raw - target_mean) / target_std
    X_v_scaled = (X_v_raw - target_mean) / target_std
    X_te_scaled = (X_te_raw - target_mean) / target_std
    
    # Re-initialize the model from scratch for Production run
    if CONFIG['model_name'] == 'MSCFormer':
        model = MSCFormer(Parameters(CONFIG['dropout']), CONFIG['num_classes'], CONFIG['num_channels']).cuda() 
    elif CONFIG['model_name'] == 'TCNet':
         model = TCNetModel(CONFIG['num_classes'], CONFIG['num_channels'], 1000, CONFIG['f1'], CONFIG['kernel_size'], CONFIG['dropout']).cuda()
    elif CONFIG['model_name'] == 'EEGEncoder':
        model = EEGEncoder(CONFIG['num_classes'], CONFIG['num_channels'], 1000).cuda()

    testAcc, Y_true, Y_pred, df_process, _ = train_subject(
        model, sub_idx, X_tr_scaled, y_tr_raw, X_v_scaled, y_v_raw, X_te_scaled, y_te, CONFIG, run_name="Production"
    )

    # Save Excel Results for Production
    df_pred_true = pd.DataFrame({'pred': Y_pred.cpu().numpy().astype(int), 'true': Y_true.cpu().numpy().astype(int)})
    save_to_excel(file_process, df_process, f"{sub_idx}_Production")
    save_to_excel(file_pred, df_pred_true, f"{sub_idx}_Production")

    # Track Final Metrics
    true_cpu = Y_true.cpu().numpy().astype(int)
    pred_cpu = Y_pred.cpu().numpy().astype(int)
    accuracy, precison, recall, f1, kappa = calMetrics(true_cpu, pred_cpu)
    production_results.append({
        'accuracy': accuracy*100, 'precision': precison*100, 
        'recall': recall*100, 'f1': f1*100, 'kappa': kappa*100
    })

    # Final Output Summaries
    df_result = pd.DataFrame(production_results)
    mean, std = df_result.mean(axis=0), df_result.std(axis=0)
    mean.name, std.name = 'mean', 'std'
    df_result = pd.concat([df_result, pd.DataFrame(mean).T, pd.DataFrame(std).T])
    
    print("\n" + "-"*15 + " FINAL PRODUCTION RESULTS " + "-"*15)
    print(df_result)
