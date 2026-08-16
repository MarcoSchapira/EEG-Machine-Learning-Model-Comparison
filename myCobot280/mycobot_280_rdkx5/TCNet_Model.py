import torch.nn as nn
from torcheeg.models import TCNet

class TCNetModel(nn.Module):
    def __init__(self, number_class, number_channel, chunk_size=1000, 
                 n_filters=8, kernel_size=64, dropout=0.5):
        super().__init__()
        
        self.model = TCNet(
            num_classes=number_class,
            num_electrodes=number_channel,
            F1=n_filters,                   
            eegnet_kernel_size=kernel_size, 
            eegnet_dropout=dropout          
        )

    def forward(self, x):
        # The data is already strictly 3D thanks to Dataset.py!
        out = self.model(x)
        
        # Return tuple to satisfy 'features, outputs = model(img)' in Train.py
        return out, out