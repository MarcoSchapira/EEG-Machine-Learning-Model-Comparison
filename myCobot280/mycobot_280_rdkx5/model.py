"""
model.py
Dynamically builds architecture based on sequence length.
Includes Adaptive Transformer, Multi-Scale 1D-CNN, 2D ResNet Spectrogram, and Multi-Head TCN-LSTM options.
Includes CLI utility to count trainable parameters based on architecture mode.
"""
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].unsqueeze(0)
        return x

class Adaptive_EMG_Model(nn.Module):
    def __init__(self, num_classes=12, num_dof=7, hidden_dim=128, num_channels=6, feature_ext='none'):
        super(Adaptive_EMG_Model, self).__init__()
        self.feature_ext = feature_ext.lower()
        # Inside Adaptive_EMG_Model.__init__
        if feature_ext == 'td':
            # This should now result in num_channels * 6
            self.input_dim = num_channels * 6
        
        if self.feature_ext in ['none', 'pca']:
            self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=32, kernel_size=5, padding=2, dilation=1)
            self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=4, dilation=2)
            
            pool_size = 5 if self.feature_ext == 'pca' else 100
            self.adaptive_pool = nn.AdaptiveAvgPool1d(pool_size)
            
            self.pos_encoder = PositionalEncoding(d_model=64, max_len=pool_size)
            encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, dropout=0.2, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            
        # Inside Adaptive_EMG_Model.__init__ in model.py
        elif self.feature_ext == 'td':
            # Update this multiplier from 4 to 6 to match your augmented dataset.py
            flat_input_size = num_channels * 6 
            self.mlp_backbone = nn.Sequential(
                nn.BatchNorm1d(flat_input_size),
                nn.Linear(flat_input_size, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, hidden_dim),
                nn.ReLU()
    )

        self.class_head = nn.Sequential(
            nn.Linear(64 if self.feature_ext in ['none', 'pca'] else hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
        self.reg_head = nn.Sequential(
            nn.Linear(64 if self.feature_ext in ['none', 'pca'] else hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_dof),
            nn.Tanh() 
        )

    def forward(self, x, apply_gating=True):
        if self.feature_ext in ['none', 'pca']:
            out = F.relu(self.conv1(x))
            out = F.relu(self.conv2(out))
            out = self.adaptive_pool(out)
            out = out.permute(0, 2, 1)
            out = self.pos_encoder(out)
            transformer_out = self.transformer(out)
            last_hidden = transformer_out.mean(dim=1)
            
        elif self.feature_ext == 'td':
            out = x.reshape(x.size(0), -1)
            last_hidden = self.mlp_backbone(out)
            
        class_logits = self.class_head(last_hidden)
        reg_preds = self.reg_head(last_hidden)
        
        if apply_gating:
            probs = F.softmax(class_logits, dim=-1)
            rest_probs = probs[:, 0]
            rest_state_scaled = torch.tensor([0.2, 0.2, -0.4, 0.2, 0.2, 0.2, -1.0], device=x.device)
            mask = (rest_probs > 0.8).unsqueeze(1)
            reg_preds = torch.where(mask, rest_state_scaled, reg_preds)
            
        return class_logits, reg_preds

class MultiScale_1D_CNN(nn.Module):
    def __init__(self, num_classes=12, num_dof=7, num_channels=6):
        super(MultiScale_1D_CNN, self).__init__()
        
        self.branch_high_freq = nn.Sequential(
            nn.Conv1d(num_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.branch_mid_freq = nn.Sequential(
            nn.Conv1d(num_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.branch_low_freq = nn.Sequential(
            nn.Conv1d(num_channels, 32, kernel_size=11, padding=5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool1d(10)
        
        self.fc = nn.Sequential(
            nn.Linear(96 * 10, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.class_head = nn.Linear(128, num_classes)
        self.reg_head = nn.Sequential(
            nn.Linear(128, num_dof),
            nn.Tanh()
        )
        
    def forward(self, x, apply_gating=True):
        b1 = self.branch_high_freq(x)
        b2 = self.branch_mid_freq(x)
        b3 = self.branch_low_freq(x)
        
        out = torch.cat([b1, b2, b3], dim=1)
        out = self.adaptive_pool(out)
        out = out.view(out.size(0), -1)
        hidden = self.fc(out)
        
        class_logits = self.class_head(hidden)
        reg_preds = self.reg_head(hidden)
        
        if apply_gating:
            probs = F.softmax(class_logits, dim=-1)
            rest_probs = probs[:, 0]
            rest_state_scaled = torch.tensor([0.2, 0.2, -0.4, 0.2, 0.2, 0.2, -1.0], device=x.device)
            mask = (rest_probs > 0.8).unsqueeze(1)
            reg_preds = torch.where(mask, rest_state_scaled, reg_preds)
            
        return class_logits, reg_preds

class ResNet_Spectrogram(nn.Module):
    def __init__(self, num_classes=12, num_dof=7, num_channels=6):
        super(ResNet_Spectrogram, self).__init__()
        
        self.resnet = models.resnet18(pretrained=True)
        
        old_conv = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(num_channels, old_conv.out_channels, 
                                      kernel_size=old_conv.kernel_size, 
                                      stride=old_conv.stride, 
                                      padding=old_conv.padding, 
                                      bias=old_conv.bias)
        
        with torch.no_grad():
            self.resnet.conv1.weight[:] = old_conv.weight.mean(dim=1, keepdim=True).repeat(1, num_channels, 1, 1)

        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        
        self.class_head = nn.Linear(num_ftrs, num_classes)
        self.reg_head = nn.Sequential(
            nn.Linear(num_ftrs, num_dof),
            nn.Tanh()
        )
        
    def forward(self, x, apply_gating=True):
        n_fft = 64
        hop_length = 16
        
        b, c, s = x.size()
        x_flat = x.view(b * c, s)
        
        window = torch.hann_window(n_fft).to(x.device)
        stft_out = torch.stft(x_flat, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
        
        spec = torch.abs(stft_out)
        spec = torch.log1p(spec) 
        
        _, f, t = spec.size()
        spec = spec.view(b, c, f, t)
        
        features = self.resnet(spec)
        
        class_logits = self.class_head(features)
        reg_preds = self.reg_head(features)
        
        if apply_gating:
            probs = F.softmax(class_logits, dim=-1)
            rest_probs = probs[:, 0]
            rest_state_scaled = torch.tensor([0.2, 0.2, -0.4, 0.2, 0.2, 0.2, -1.0], device=x.device)
            mask = (rest_probs > 0.8).unsqueeze(1)
            reg_preds = torch.where(mask, rest_state_scaled, reg_preds)
            
        return class_logits, reg_preds

class MultiHeadTCN_LSTM(nn.Module):
    def __init__(self, num_classes=12, num_dof=7, num_channels=6):
        super(MultiHeadTCN_LSTM, self).__init__()
        
        # Multi-Head TCN (Temporal Convolutional Network)
        # Replaces raw feature math with learned high/mid/low frequency extraction
        self.branch_high_freq = nn.Sequential(
            nn.Conv1d(num_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5) # Compresses sequence from 500 to 100 steps
        )
        self.branch_mid_freq = nn.Sequential(
            nn.Conv1d(num_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5)
        )
        self.branch_low_freq = nn.Sequential(
            nn.Conv1d(num_channels, 32, kernel_size=11, padding=5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5)
        )
        
        # LSTM Backbone
        # Input features: 32 channels * 3 branches = 96
        self.lstm = nn.LSTM(input_size=96, hidden_size=128, num_layers=2, batch_first=True, dropout=0.3)
        
        # Dual Output Heads
        self.class_head = nn.Linear(128, num_classes)
        self.reg_head = nn.Sequential(
            nn.Linear(128, num_dof),
            nn.Tanh()
        )
        
    def forward(self, x, apply_gating=True):
        # 1. Multi-Head Frequency Filtering
        b1 = self.branch_high_freq(x)
        b2 = self.branch_mid_freq(x)
        b3 = self.branch_low_freq(x)
        
        # Concatenate branches along channel dimension (Batch, 96, 100)
        tcn_out = torch.cat([b1, b2, b3], dim=1)
        
        # 2. Sequence Mapping
        # LSTM expects (Batch, Sequence, Features), so permute channels and sequence
        tcn_out = tcn_out.permute(0, 2, 1)
        
        lstm_out, (h_n, c_n) = self.lstm(tcn_out)
        
        # 3. Final Prediction
        # Slice out the final chronological time step from the LSTM
        final_step = lstm_out[:, -1, :] 
        
        class_logits = self.class_head(final_step)
        reg_preds = self.reg_head(final_step)
        
        if apply_gating:
            probs = F.softmax(class_logits, dim=-1)
            rest_probs = probs[:, 0]
            rest_state_scaled = torch.tensor([0.2, 0.2, -0.4, 0.2, 0.2, 0.2, -1.0], device=x.device)
            mask = (rest_probs > 0.8).unsqueeze(1)
            reg_preds = torch.where(mask, rest_state_scaled, reg_preds)
            
        return class_logits, reg_preds

def get_model(model_type, num_channels, feature_ext='none', num_classes=12):
    if model_type == 'adaptive':
        return Adaptive_EMG_Model(num_classes=num_classes, num_channels=num_channels, feature_ext=feature_ext)
    elif model_type == 'multiscale_cnn':
        return MultiScale_1D_CNN(num_classes=num_classes, num_channels=num_channels)
    elif model_type == 'resnet_spectrogram':
        return ResNet_Spectrogram(num_classes=num_classes, num_channels=num_channels)
    elif model_type == 'tcn_lstm':
        return MultiHeadTCN_LSTM(num_classes=num_classes, num_channels=num_channels)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")
    return total_params

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count_params", action="store_true")
    parser.add_argument("--model_type", type=str, choices=['adaptive', 'multiscale_cnn', 'resnet_spectrogram', 'tcn_lstm'], default='adaptive')
    parser.add_argument("--electrodes", nargs='+', type=int, default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--feature_ext", type=str, choices=['none', 'pca', 'td'], default='none')
    args = parser.parse_args()
    
    num_channels = len(args.electrodes)
    try:
        model = get_model(args.model_type, num_channels, args.feature_ext)
        print(f"Model initialized: {args.model_type.upper()} with {num_channels} channels.")
    except Exception as e:
        print(f"Error initializing model: {e}")
        exit(1)
        
    if args.count_params:
        count_parameters(model)