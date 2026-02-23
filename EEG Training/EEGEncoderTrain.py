import numpy as np
import torch
from torch.utils import data
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch.nn as nn
import torch.nn.functional as F
import pickle
import random
import tqdm
from sklearn.metrics import accuracy_score, recall_score, precision_score, cohen_kappa_score
from transformers import LlamaConfig
from Lama_for_EEGEncoder import LlamaForCausalLM
import tqdm
import os
import warnings
import transformers

# 1. Silence Python Warnings
# This stops the "FutureWarning: torch.cuda.amp..." and "UserWarning: padding='same'..."
warnings.filterwarnings("ignore")

# 2. Silence Hugging Face Transformers
# This stops the massive "LlamaForCausalLM has generative capabilities..." block
transformers.logging.set_verbosity_error()


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'

all_len = 1000

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


setup_seed(32)


class CustomEEGDB(data.Dataset):
    def __init__(self, subject_id, mode='train', data_dir='D:\EEG Data\EEG_11\EEG_Compact\Processed_Train123Test3', noise_std=0.05, make_easier = True):
        """
        subject_id: int (1 to 25)
        mode: 'train' or 'val'
        data_dir: path to your npz files
        """
        self.mode = mode
        self.noise_std = noise_std
        self.make_easier = make_easier
        # 1. Construct the filename (e.g., "C01T.npz" or "C01V.npz")
        # {:02d} ensures that 1 becomes '01', 2 becomes '02', etc.
        suffix = 'T' if mode == 'train' else 'V'
        filename = f"C{subject_id:02d}{suffix}.npz"
        file_path = os.path.join(data_dir, filename)
        
        # 2. Load the npz file
        print(f"Loading {file_path}...")
        loaded_data = np.load(file_path)
        
        # Extract data and labels
        # Assuming data shape is (trials, channels, time) -> (N, 60, 1000)
        self.x = loaded_data['data'] 
        self.y_raw = loaded_data['labels']
        
        # 3. Standardization (Z-score normalization)
        # We need to reshape to (N, Ch * Time) to fit scaler, then reshape back
        # This gives mean=0, var=1 for each channel across the dataset
        N, C, T = self.x.shape
        scaler = StandardScaler()
        
        # Reshape to (Trials, Channels * Time) for scaling or (Trials*Time, Channels) 
        # The standard BCI approach is usually scaling each channel independently
        x_reshaped = self.x.transpose(0, 2, 1).reshape(-1, C) # (N*T, C)
        x_scaled = scaler.fit_transform(x_reshaped)
        self.x = x_scaled.reshape(N, T, C).transpose(0, 2, 1) # Back to (N, C, T)
        
        # Reshape from (N, 60, 1000) -> (N, 1, 60, 1000)
        self.x = self.x[:, np.newaxis, :, :]

        # 4. Handle Labels (Map trigger numbers to 0-11)
        # Note: Ideally, you fit the LabelEncoder on training and transform validation
        # But if both sets have all 12 classes, this works per file.
        self.y_encoded = self.y_raw.flatten().astype(int) - 1
        
        if self.make_easier == True:
            # Current mapping (0-11):
            # 6, 8, 9 -> Grasping -> Group to 6
            # 7 -> Rest -> Keep at 7
            # 10, 11 -> Twist -> Group to 8
            
            # Create a copy to avoid overwriting issues during assignment
            new_labels = np.copy(self.y_encoded)

            # Group Grasping (Old 6, 8, 9 become New 6)
            mask_grasp = np.isin(self.y_encoded, [6, 8, 9])
            new_labels[mask_grasp] = 6

            # Group Twisting (Old 10, 11 become New 8)
            mask_twist = np.isin(self.y_encoded, [10, 11])
            new_labels[mask_twist] = 8
            
            # Note: Old 7 (Rest) stays 7, so we don't need to touch it.
            self.y_encoded = new_labels
            
            # Update class count to 9
            self.n_classes = 9 

            # Verify labels are in range [0, 8]
            if np.min(self.y_encoded) < 0 or np.max(self.y_encoded) > 8:
                raise ValueError(f"Labels out of range! Found min: {np.min(self.y_encoded)} max: {np.max(self.y_encoded)}")
        
        # Convert to One-Hot Encoding for the model
        self.y_onehot = np.eye(self.n_classes)[self.y_encoded]
        
        # Convert to Tensor
        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.y = torch.tensor(self.y_onehot, dtype=torch.float32)

        print(f"Loaded Subject {subject_id} ({mode}): Data {self.x.shape}, Labels {self.y.shape}")

    def add_gaussian_noise(self, tensor):
        """
        Adds random noise to the signal. 
        std=0.05 means noise is 5% of the signal's standard deviation.
        """
        noise = torch.randn(tensor.shape) * self.noise_std
        return tensor + noise

    def __getitem__(self, index):
        x_sample = self.x[index]
        y_sample = self.y[index]

        # Apply noise ONLY if we are in training mode
        if self.mode == 'train':
            x_sample = self.add_gaussian_noise(x_sample)
            
        return x_sample, y_sample

    def __len__(self):
        return len(self.x)
    

class MixUp:
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, data, target):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        batch_size = data.size()[0]
        index = torch.randperm(batch_size)

        mixed_data = lam * data + (1 - lam) * data[index, :]
        target_a, target_b = target, target[index]
        return mixed_data, target_a, target_b, lam

    def loss_func(self, pred, target_a, target_b, lam):
        return lam * torch.nn.functional.cross_entropy(pred, target_a, label_smoothing=0.1) + (
                1 - lam) * torch.nn.functional.cross_entropy(
            pred, target_b, label_smoothing=0.1)

class LinearL2(nn.Module):
    def __init__(self, in_features, out_features, weight_decay=0.):
        super(LinearL2, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.weight_decay = weight_decay

    def forward(self, x):
        # 在前向传播中，我们不做任何关于权重衰减的事情
        return self.linear(x)

    def l2_loss(self):
        # 计算 L2 损失，这将在训练循环中使用
        return self.weight_decay * torch.sum(self.linear.weight ** 2)


class Conv1dL2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, weight_decay=0., bias=False):
        super(Conv1dL2, self).__init__()
        self.conv1 = nn.Conv1d(in_channels,
                               out_channels,
                               kernel_size,
                               stride=stride,
                               padding=padding,
                               dilation=dilation,
                               bias=bias,
                               groups=groups)
        self.weight_decay = weight_decay

    def forward(self, x):
        # 在前向传播中，我们不做任何关于权重衰减的事情
        return self.conv1(x)

    def l2_loss(self):
        # 计算 L2 损失，这将在训练循环中使用
        return self.weight_decay * torch.sum(self.conv1.weight ** 2)


class Conv2dL2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, weight_decay=0., bias=False):
        super(Conv2dL2, self).__init__()
        self.conv2 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size,
                               stride=stride,
                               padding=padding,
                               dilation=dilation,
                               bias=bias,
                               groups=groups,
                               )
        self.weight_decay = weight_decay

    def forward(self, x):
        # 在前向传播中，我们不做任何关于权重衰减的事情
        return self.conv2(x)

    def l2_loss(self):
        # 计算 L2 损失，这将在训练循环中使用
        return self.weight_decay * torch.sum(self.conv2.weight ** 2)


class ConvBlock(nn.Module):
    def __init__(self, F1=4, kernLength=64, poolSize=8, D=2, in_chans=22, dropout=0.5):
        super(ConvBlock, self).__init__()
        F2 = F1 * D
        # self.conv1 = nn.Conv2d(1, F1, (kernLength, 1), padding='same', bias=False)
        self.conv1 = Conv2dL2(1, F1, (kernLength, 1), padding='same', bias=False, weight_decay=0.009)

        self.batchnorm1 = nn.BatchNorm2d(F1)
        # self.depthwise = nn.Conv2d(F1, F1 * D, (1, in_chans), groups=F1, bias=False)
        self.depthwise = Conv2dL2(F1, F1 * D, (1, in_chans), groups=F1, bias=False, weight_decay=0.009)
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)
        self.activation = nn.ELU()
        self.avgpool1 = nn.AvgPool2d((8, 1))
        self.dropout1 = nn.Dropout(dropout)
        # self.conv2 = nn.Conv2d(F1 * D, F2, (16, 1), padding='same', bias=False)
        self.conv2 = Conv2dL2(F1 * D, F2, (16, 1), padding='same', bias=False, weight_decay=0.009)
        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.avgpool2 = nn.AvgPool2d((poolSize, 1))
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 1, 3, 2)
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.depthwise(x)
        x = self.batchnorm2(x)
        x = self.activation(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.batchnorm3(x)
        x = self.activation(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)
        return x


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttentionBlock, self).__init__()
        # self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.3)
        self.dp = nn.Dropout(0.3)

        model_cofig1 = LlamaConfig()
        model_cofig1.hidden_size = embed_dim
        model_cofig1.pad_token_id = 0
        model_cofig1.intermediate_size = embed_dim * 1
        model_cofig1.num_hidden_layers = 2
        model_cofig1.num_attention_heads = num_heads
        model_cofig1.vocab_size = 21
        model_cofig1.max_position_embeddings = 500
        model_cofig1.type_vocab_size = 20
        model_cofig1.dropout_ratio = 0.3
        model_cofig1.weight_decay = 0.5
        # model_cofig1.initializer_range = 0.1
        self.short_encoder = LlamaForCausalLM(config=model_cofig1)

    def forward(self, x):
        x0 = x
        # MultiheadAttention in PyTorch expects inputs of shape (L, N, E)
        # where L is the sequence length, N is the batch size, and E is the embedding dimension.
        # out, _ = self.mha(x, x, x)

        out = self.short_encoder(inputs_embeds=x, output_hidden_states=True).hidden_states[-1]
        out = self.dp(x0 + out)
        return out  # Permute back to (N, L, E)

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=2, residual=False, apply_to_input=True):
        super(AttentionBlock, self).__init__()
        self.residual = residual
        self.apply_to_input = apply_to_input
        self.attention = MultiHeadAttentionBlock(embed_dim, num_heads)


    def forward(self, x):
        out = self.attention(x)

        return out

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TCNBlock_(nn.Module):
    def __init__(self, input_dimension, depth, kernel_size, filters, dropout, weight_decay=0.009, max_norm=0.6, activation='relu'):
        super(TCNBlock_, self).__init__()
        self.depth = depth
        self.activation = getattr(F, activation)
        self.dropout = dropout
        self.blocks = nn.ModuleList()
        self.downsample = nn.Conv1d(input_dimension, filters, 1) if input_dimension != filters else None
        self.cn1 = nn.Sequential(Conv1dL2(input_dimension, filters, kernel_size, weight_decay=0.009), nn.BatchNorm1d(filters), nn.SiLU(),nn.Dropout(0.3))
        self.cn2 = nn.Sequential(Conv1dL2(filters, filters, kernel_size, weight_decay=0.009), nn.BatchNorm1d(filters), nn.SiLU(), nn.Dropout(0.3))



        for i in range(depth-1):
            dilation_size = 2 ** (i+1)
            padding = (kernel_size - 1) * dilation_size
            block_layers = [
                Conv1dL2(filters if i > 0 else input_dimension, filters, kernel_size, stride=1, padding=padding, dilation=dilation_size,
                                     weight_decay=0.009),
                Chomp1d(padding),
                nn.BatchNorm1d(filters),
                nn.ReLU(),
                nn.Dropout(dropout),
                Conv1dL2(filters, filters, kernel_size, stride=1, padding=padding, dilation=dilation_size, weight_decay=0.009),
                Chomp1d(padding),
                nn.BatchNorm1d(filters),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            self.blocks.append(nn.Sequential(*block_layers))

        self.init_weights(max_norm)

    def init_weights(self, max_norm):
        for block in self.blocks:
            for layer in block:
                if isinstance(layer, nn.Conv1d):
                    layer.weight.data = nn.init.kaiming_uniform_(layer.weight.data)
                    nn.utils.clip_grad_norm_(layer.parameters(), max_norm)

    def forward(self, x):
        out = x.transpose(1, 2)
        out = self.cn1(out)
        out = self.cn2(out)
        res = self.downsample(out) if self.downsample is not None else out

        for i, block in enumerate(self.blocks):
            if i == 0:
                out = block(out)
                out += res
            else:
                out = block(out)
                out += self.blocks[i-1](res)
            out = self.activation(out)

        return out.transpose(1, 2)

class EEGEncoder(nn.Module):
    def __init__(self, n_classes=4, in_chans=22, in_samples=1125, n_windows=5, attention='mha',
                 eegn_F1=16, eegn_D=2, eegn_kernelSize=64, eegn_poolSize=7, eegn_dropout=0.3,
                 tcn_depth=2, tcn_kernelSize=4, tcn_filters=32, tcn_dropout=0.3,
                 tcn_activation='elu', fuse='average'):
        super(EEGEncoder, self).__init__()
        self.n_windows = n_windows
        self.fuse = fuse
        self.dense_weight_decay = 0.5
        self.from_logits = True

        F2 = eegn_F1 * eegn_D

        # self.conv_block = ConvBlock(eegn_F1, eegn_F1, eegn_D, eegn_kernelSize, eegn_poolSize, eegn_dropout)
        self.conv_block = ConvBlock(F1=eegn_F1, kernLength=eegn_kernelSize, poolSize=7, D=2, in_chans=22, dropout=eegn_dropout)
        #self.attention_block = AttentionBlock(embed_dim=F2, num_heads=4)  # Define your attention block
        self.tcn_blocks = nn.ModuleList([TCNBlock_(F2, tcn_depth, tcn_kernelSize, tcn_filters, tcn_dropout, tcn_activation) for _ in range(n_windows)])
        self.dense_layers = nn.ModuleList([LinearL2(tcn_filters, n_classes, 0.5) for _ in range(n_windows)])
        self.aa_drop = nn.Dropout(0.3)
        model_cofig1 = LlamaConfig()
        model_cofig1.hidden_size = F2
        model_cofig1.pad_token_id = 0
        model_cofig1.intermediate_size = F2 * 1
        model_cofig1.num_hidden_layers = 2
        model_cofig1.num_attention_heads = 2
        model_cofig1.vocab_size = 21
        model_cofig1.max_position_embeddings = 500
        model_cofig1.type_vocab_size = 20
        model_cofig1.dropout_ratio = 0.3
        model_cofig1.weight_decay = 0.5
        # model_cofig1.initializer_range = 0.1
        self.trm_block = nn.ModuleList([LlamaForCausalLM(config=model_cofig1) for _ in range(n_windows)])


        if fuse == 'concat':
            self.final_dense = LinearL2(n_classes * n_windows, n_classes, 0.5)

    def forward(self, x):
        x = self.conv_block(x)
        x = x[:, :, :, 0].permute(0, 2, 1)  # Equivalent to Lambda(lambda x: x[:,:,-1,:])(block1)
        sw_outputs = []
        for i in range(self.n_windows):
            # st = i
            # end = x.shape[1] - self.n_windows + i + 1
            window_slice = self.aa_drop(x[:, :, :])
            # Apply attention if defined
            # window_slice = self.attention_block(window_slice)
            # Apply TCN block
            tcn_output = self.tcn_blocks[i](window_slice)
            tcn_output = tcn_output[:, -1, :]  # Equivalent to Lambda(lambda x: x[:,-1,:])(block3)

            trm_output = self.trm_block[i](inputs_embeds=window_slice, output_hidden_states=True).hidden_states[-1].mean(1)
            tcn_output = tcn_output+F.dropout(trm_output, 0.3)
            # window_slice = window_slice-x.mean(1, keepdim=True)
            # tcn_output = self.trm_block[i](inputs_embeds=window_slice, output_hidden_states=True).hidden_states[-1]
            # tcn_output = F.relu(F.dropout(tcn_output.mean(1), 0.3))
            # tcn_output = window_slice.mean(1)
            # Apply dense layer
            dense_output = self.dense_layers[i](tcn_output)
            sw_outputs.append(dense_output)

        if self.fuse == 'average':
            out = torch.mean(torch.stack(sw_outputs, dim=0), dim=0)
        elif self.fuse == 'concat':
            out = torch.cat(sw_outputs, dim=1)
            out = self.final_dense(out)

        if not self.from_logits:
            out = F.softmax(out, dim=1)

        return None, out


if __name__ == '__main__':
    
    # Configuration for your new data
    make_easier = True
    if make_easier == True:
        n_classes = 9
    else:
        n_classes = 12
    in_chans = 60
    in_samples = 1000
    n_subjects = 25
    data_path = 'D:\EEG Data\EEG_11\EEG_Compact\Processed_Train123Test3'
    
    final_res_lst = []
    # Note: removed cuda.amp.GradScaler for simplicity unless you are sure you need mixed precision
    # If your GPU is older, mixed precision might cause instability. 
    scaler = torch.cuda.amp.GradScaler() 

    # Loop from 1 to 25
    for sub_id in range(1, n_subjects + 1): 
        
        epoch = 500
        bs = 64
        
        eeg_model = EEGEncoder(n_classes=n_classes, in_chans=in_chans, in_samples=in_samples).to('cuda')
        
        try:
            train_db = CustomEEGDB(subject_id=sub_id, mode='train', data_dir=data_path, make_easier=make_easier)
            val_db = CustomEEGDB(subject_id=sub_id, mode='val', data_dir=data_path, make_easier=make_easier)
        except FileNotFoundError as e:
            print(f"Skipping Subject {sub_id}: File not found ({e})")
            continue

        optimizer = torch.optim.Adam(eeg_model.parameters(), lr=1e-3)
        
        train_loader = torch.utils.data.DataLoader(train_db, batch_size=bs, num_workers=0, shuffle=True, drop_last=False)
        val_loader = torch.utils.data.DataLoader(val_db, batch_size=bs, num_workers=0, shuffle=False, drop_last=False)

        best_acc = 0
        best_acc0 = 0
        loop = tqdm.tqdm(range(epoch))
        
        class_weights = torch.ones(n_classes).to('cuda') * 11.0  # Set all to 11
        class_weights[7] = 1.0  # Set Rest (index 7) to 1
        
        # 2. Pass weights to the loss function
        loss_func = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.2)

        for e in loop:
            # --- TRAIN LOOP ---
            eeg_model.train()
            label_lst = []
            outs_lst = []
            
            for inputs, labels in train_loader:
                inputs = inputs.to('cuda')
                labels = labels.to('cuda')

                optimizer.zero_grad()
                
                with torch.cuda.amp.autocast():
                    outs = eeg_model(inputs)
                    loss = loss_func(outs, labels)
                    
                    # Calculate L2 loss from model modules
                    l2_loss = sum(module.l2_loss() for name, module in eeg_model.named_modules() if hasattr(module, 'l2_loss'))
                
                scaler.scale(2 * (loss + l2_loss)).backward()
                scaler.step(optimizer)
                scaler.update()

                outs_lst.extend(outs.argmax(-1).cpu().detach().numpy().tolist())
                label_lst.extend(labels.argmax(-1).cpu().detach().numpy().tolist())
            
            train_acc = np.round(accuracy_score(label_lst, outs_lst), 4)

            # --- VALIDATION LOOP ---
            eeg_model.eval()
            label_lst = []
            outs_lst0 = []
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to('cuda')
                    # labels are already one-hot, but for accuracy calculation we need indices
                    
                    outs = eeg_model(inputs)
                    outs_lst0.extend(outs.argmax(-1).cpu().detach().numpy().tolist())
                    label_lst.extend(labels.argmax(-1).cpu().detach().numpy().tolist())

            val_acc = np.round(accuracy_score(label_lst, outs_lst0), 4)
            kappa = np.round(cohen_kappa_score(label_lst, outs_lst0), 4)

            # --- LOGGING ---
            if val_acc > best_acc:
                best_acc = val_acc
                res_str = f'Subject {sub_id} | Epoch {e} | Best Val Acc: {best_acc} | Kappa: {kappa}'
                print(res_str)
                # Optional: Save model
                # torch.save(eeg_model.state_dict(), f'model_sub{sub_id}.pth')

            loop.set_postfix(Epoch=e, Train_Acc=train_acc, Val_Acc=val_acc, Best_Val=best_acc)
        
        final_res_lst.append(f'Subject {sub_id}: {best_acc}')

    print("Final Results per Subject:")
    print(final_res_lst)