import torch.nn as nn

# ==========================================
# 1. BASELINE MODEL
# ==========================================
class SanityCheckCNN(nn.Module):
    """
    A minimal 1-layer 1D CNN used to verify that the pipeline, 
    data loaders, and training loops are functioning correctly.
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=60, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        
        # Global Pooling squashes the variable time steps down to 1
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier Head (Binary Classification = 2 outputs)
        self.classifier = nn.Linear(32, 2)

    def forward(self, x):
        # x input shape: [Batch, 1, 60, Time_Steps]
        x = x.squeeze(1)            # -> [Batch, 60, Time_Steps]
        
        x = self.relu(self.conv(x)) # -> [Batch, 32, Time_Steps]
        x = self.global_pool(x)     # -> [Batch, 32, 1]
        x = x.view(x.size(0), -1)   # -> [Batch, 32]
        
        return self.classifier(x)   # -> [Batch, 2]


# ==========================================
# 2. DEEP CNN
# ==========================================
# class CNN1D3(nn.Module):
#     """
#     A deeper, 3-layer 1D CNN architecture with max pooling 
#     and dropout for improved generalization on audio features.
#     """
#     def __init__(self):
#         super().__init__()
        
#         # Block 1
#         self.conv1 = nn.Conv1d(in_channels=60, out_channels=32, kernel_size=3, padding=1)
#         self.relu1 = nn.ReLU()
#         self.pool1 = nn.MaxPool1d(kernel_size=2)

#         # Block 2
#         self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
#         self.relu2 = nn.ReLU()
#         self.pool2 = nn.MaxPool1d(kernel_size=2)

#         # Block 3
#         self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
#         self.relu3 = nn.ReLU()

#         # Global Pooling & Fully Connected Layers
#         self.global_pool = nn.AdaptiveAvgPool1d(1)
        
#         self.fc1 = nn.Linear(128, 64)
#         self.dropout = nn.Dropout(p=0.5)
#         self.fc2 = nn.Linear(64, 2)

#     def forward(self, x):
#         # x input shape: [Batch, 1, 60, Time_Steps]
#         x = x.squeeze(1)
        
#         # Feature Extraction
#         x = self.pool1(self.relu1(self.conv1(x)))
#         x = self.pool2(self.relu2(self.conv2(x)))
#         x = self.relu3(self.conv3(x))
        
#         # Pooling
#         x = self.global_pool(x)
#         x = x.view(x.size(0), -1)
        
#         # Classification
#         x = self.relu1(self.fc1(x))
#         x = self.dropout(x)
#         return self.fc2(x)


class CNN1D3(nn.Module):
    """
    3-layer 1D CNN for 6-class language classification using MFCC features.
    Input shape: [batch, 1, 60, time_steps]
    """
    def __init__(self, num_classes=6):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels=60, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc1 = nn.Linear(128, 64)
        self.relu_fc = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: [batch, 1, 60, time_steps]
        x = x.squeeze(1)  # [batch, 60, time_steps]
        
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.relu3(self.conv3(x))
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        x = self.relu_fc(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x