import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, F1=4, kernLength=64, poolSize=8, D=2, in_chans=22, dropout=0.5):
        super(ConvBlock, self).__init__()
        F2 = F1 * D
        
        # Swapped custom Conv2dL2 for standard nn.Conv2d
        self.conv1 = nn.Conv2d(1, F1, (kernLength, 1), padding='same', bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1)
        
        self.depthwise = nn.Conv2d(F1, F1 * D, (1, in_chans), groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)
        self.activation = nn.ELU()
        
        self.avgpool1 = nn.AvgPool2d((8, 1))
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv2d(F1 * D, F2, (16, 1), padding='same', bias=False)
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

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TCNBlock_(nn.Module):
    def __init__(self, input_dimension, depth, kernel_size, filters, dropout, max_norm=0.6, activation='relu'):
        super(TCNBlock_, self).__init__()
        self.depth = depth
        self.activation = getattr(F, activation)
        self.dropout = dropout
        self.blocks = nn.ModuleList()
        self.downsample = nn.Conv1d(input_dimension, filters, 1) if input_dimension != filters else None

        # Swapped custom Conv1dL2 for standard nn.Conv1d
        self.cn1 = nn.Sequential(nn.Conv1d(input_dimension, filters, kernel_size, padding='same'), nn.BatchNorm1d(filters), nn.SiLU(), nn.Dropout(0.3))
        self.cn2 = nn.Sequential(nn.Conv1d(filters, filters, kernel_size, padding='same'), nn.BatchNorm1d(filters), nn.SiLU(), nn.Dropout(0.3))

        for i in range(depth-1):
            dilation_size = 2 ** (i+1)
            padding = (kernel_size - 1) * dilation_size
            block_layers = [
                nn.Conv1d(filters if i > 0 else input_dimension, filters, kernel_size, stride=1, padding=padding, dilation=dilation_size),
                Chomp1d(padding),
                nn.BatchNorm1d(filters),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(filters, filters, kernel_size, stride=1, padding=padding, dilation=dilation_size),
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
        
        # 1. FIXED: Outputs raw logits for CrossEntropyLoss/FocalLoss
        self.from_logits = True 

        F2 = eegn_F1 * eegn_D

        self.conv_block = ConvBlock(F1=eegn_F1, kernLength=eegn_kernelSize, poolSize=eegn_poolSize, D=eegn_D, in_chans=in_chans, dropout=eegn_dropout)
        
        self.tcn_blocks = nn.ModuleList([TCNBlock_(F2, tcn_depth, tcn_kernelSize, tcn_filters, tcn_dropout, activation=tcn_activation) for _ in range(n_windows)])
        
        # Swapped LinearL2 for standard nn.Linear
        self.dense_layers = nn.ModuleList([nn.Linear(tcn_filters, n_classes) for _ in range(n_windows)])
        self.aa_drop = nn.Dropout(0.3)
        
        # 2. FIXED: Bidirectional PyTorch Transformer (Replaces Causal Llama block)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=F2, 
            nhead=2, 
            dim_feedforward=F2 * 2, 
            dropout=0.3, 
            batch_first=True
        )
        self.trm_block = nn.ModuleList([nn.TransformerEncoder(encoder_layer, num_layers=2) for _ in range(n_windows)])

        if fuse == 'concat':
            self.final_dense = nn.Linear(n_classes * n_windows, n_classes)

    def forward(self, x):
        x = self.conv_block(x)
        x = x[:, :, :, 0].permute(0, 2, 1)  # [Batch, Time, Features]
        
        sw_outputs = []
        seq_len = x.shape[1]
        
        # Calculate dynamic window sizes based on the pooled sequence length
        window_size = seq_len // self.n_windows
        
        for i in range(self.n_windows):
            # 3. FIXED: Properly implemented sliding windows
            st = i * window_size
            end = (i + 1) * window_size if i < self.n_windows - 1 else seq_len
            
            window_slice = self.aa_drop(x[:, st:end, :])
            
            # TCN Pathway
            tcn_output = self.tcn_blocks[i](window_slice)
            tcn_output = tcn_output[:, -1, :] 
            
            # Transformer Pathway (Bidirectional)
            trm_output = self.trm_block[i](window_slice)
            trm_output_pooled = trm_output.mean(dim=1)
            
            # Fuse pathways
            combined_features = tcn_output + F.dropout(trm_output_pooled, p=0.3, training=self.training)
            
            # Classification
            dense_output = self.dense_layers[i](combined_features)
            sw_outputs.append(dense_output)

        if self.fuse == 'average':
            out = torch.mean(torch.stack(sw_outputs, dim=0), dim=0)
        elif self.fuse == 'concat':
            out = torch.cat(sw_outputs, dim=1)
            out = self.final_dense(out)

        if not self.from_logits:
            out = F.softmax(out, dim=1)

        # Return features and logits to match the output tuple expected by train.py
        return combined_features, out