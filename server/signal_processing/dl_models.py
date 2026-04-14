import torch
import torch.nn as nn
import torch.nn.functional as F

class RCNN(nn.Module):
    def __init__(self, input_size=187, num_classes=5):
        super(RCNN, self).__init__()

        # Define Convolution Layers
        self.conv1_1 = nn.Conv1d(in_channels=1, out_channels=48, kernel_size=6, padding='same')
        self.bn1_1 = nn.BatchNorm1d(48)

        self.conv1_2 = nn.Conv1d(48, 64, 6, padding='same')
        self.bn1_2 = nn.BatchNorm1d(64)
        
        self.conv1_3 = nn.Conv1d(64, 32, 6, padding='same')
        self.bn1_3 = nn.BatchNorm1d(32)

        self.residual_1 = nn.Conv1d(32, 16, 16, padding='same')
        self.bn_residual_1 = nn.BatchNorm1d(16)
        
        self.conv1_4 = nn.Conv1d(48, 64, 6, padding='same')
        self.bn1_4 = nn.BatchNorm1d(64)

        self.conv1_5 = nn.Conv1d(64, 32, 6, padding='same')
        self.bn1_5 = nn.BatchNorm1d(32)

        self.residual_2 = nn.Conv1d(32, 16, 16, padding='same')
        self.bn_residual_2 = nn.BatchNorm1d(16)
        
        self.conv1_6 = nn.Conv1d(48, 64, 6, padding='same')
        self.bn1_6 = nn.BatchNorm1d(64)

        self.conv1_7 = nn.Conv1d(64, 32, 6, padding='same')
        self.bn1_7 = nn.BatchNorm1d(32)

        self.residual_3 = nn.Conv1d(32, 16, 16, padding='same')
        self.bn_residual_3 = nn.BatchNorm1d(16)
        
        self.conv1_8 = nn.Conv1d(48, 64, 6, padding='same')
        self.bn1_8 = nn.BatchNorm1d(64)

        self.conv1_9 = nn.Conv1d(64, 32, 6, padding='same')
        self.bn1_9 = nn.BatchNorm1d(32)

        self.residual_4 = nn.Conv1d(32, 16, 16, padding='same')
        self.bn_residual_4 = nn.BatchNorm1d(16)

        # Calculate the output size after convolutions and pooling
        dummy_input = torch.zeros(1, 1, input_size)  # Batch size = 1, channels = 1, input_size = input_size
        dummy_output = self._forward_convs(dummy_input)
        flattened_size = dummy_output.numel()  # Flattened size of the tensor after convolution and pooling
        # Final fully connected layers
        self.fc1 = nn.Linear(flattened_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def _forward_convs(self, x):
        # Forward pass through convolutional layers
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = F.relu(x)

        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = F.relu(x)

        x = self.conv1_3(x)
        x = self.bn1_3(x)
        x = F.relu(x)

        residual_1 = self.residual_1(x)
        residual_1 = self.bn_residual_1(residual_1)
        concat_1 = torch.cat((x, residual_1), dim=1)
        concat_1 = F.relu(concat_1)
        x = F.max_pool1d(concat_1, 6, padding=3)

        x = self.conv1_4(x)
        x = self.bn1_4(x)
        x = F.relu(x)

        x = self.conv1_5(x)
        x = self.bn1_5(x)
        x = F.relu(x)

        residual_2 = self.residual_2(x)
        residual_2 = self.bn_residual_2(residual_2)
        concat_2 = torch.cat((x, residual_2), dim=1)
        concat_2 = F.relu(concat_2)
        x = F.max_pool1d(concat_2, 6, stride=2, padding=3)

        x = self.conv1_6(x)
        x = self.bn1_6(x)
        x = F.relu(x)

        x = self.conv1_7(x)
        x = self.bn1_7(x)
        x = F.relu(x)

        residual_3 = self.residual_3(x)
        residual_3 = self.bn_residual_3(residual_3)
        concat_3 = torch.cat((x, residual_3), dim=1)
        concat_3 = F.relu(concat_3)
        x = F.max_pool1d(concat_3, 6, padding=3)

        x = self.conv1_8(x)
        x = self.bn1_8(x)
        x = F.relu(x)

        x = self.conv1_9(x)
        x = self.bn1_9(x)
        x = F.relu(x)

        residual_4 = self.residual_4(x)
        residual_4 = self.bn_residual_4(residual_4)
        concat_4 = torch.cat((x, residual_4), dim=1)
        concat_4 = F.relu(concat_4)
        x = F.max_pool1d(concat_4, 6, padding=3)

        return x

    def forward(self, x):
        # Use the _forward_convs function for convolutional layers
        x = self._forward_convs(x)

        # Flatten the output from convolution layers before passing to fully connected layers
        flat = x.view(x.size(0), -1)
        x = self.fc1(flat)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        
        return x
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureExtractionBlock(nn.Module):
    def __init__(self, in_channels, out_channels_deep=16, out_channels_shallow=6):
        super(FeatureExtractionBlock, self).__init__()
        # Cascaded convolutional layers (Deep features) [cite: 126]
        self.deep_path = nn.Sequential(
            nn.Conv1d(in_channels, out_channels_deep, kernel_size=32, padding='same'),
            nn.ReLU(),
            nn.Conv1d(out_channels_deep, out_channels_deep, kernel_size=32, padding='same'),
            nn.ReLU()
        )
        # Parallel convolutional layer (Shallow features) 
        self.shallow_path = nn.Sequential(
            nn.Conv1d(in_channels, out_channels_shallow, kernel_size=16, padding='same'),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool1d(kernel_size=2) # 

    def forward(self, x):
        deep = self.deep_path(x)
        shallow = self.shallow_path(x)
        # Concatenate deep and shallow features 
        combined = torch.cat((deep, shallow), dim=1)
        return self.maxpool(combined)

class ECGClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super(ECGClassifier, self).__init__()
        
        # 3x Feature Extraction Blocks 
        # Input size: (Batch, 1, 187) 
        self.block1 = FeatureExtractionBlock(1, 16, 6) # Output channels: 22
        self.block2 = FeatureExtractionBlock(22, 16, 6)
        self.block3 = FeatureExtractionBlock(22, 16, 6)
        
        # Multi-Head Attention [cite: 125, 127]
        # After 3 blocks (187 / 2 / 2 / 2) ≈ 23 time steps
        self.attention = nn.MultiheadAttention(embed_dim=22, num_heads=2, batch_first=True)
        
        # GRU Layers [cite: 124, 127]
        self.gru = nn.GRU(input_size=22, hidden_size=32, num_layers=1, batch_first=True)
        
        # Fully Connected Stage 
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, 32)
        self.out = nn.Linear(32, num_classes) # Softmax applied in loss or during inference 

    def forward(self, x):
        # x shape: (Batch, 1, 187)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Reshape for Attention: (Batch, Seq_Len, Features)
        x = x.permute(0, 2, 1)
        
        # Multi-Head Attention 
        attn_output, _ = self.attention(x, x, x)
        
        # GRU 
        gru_out, _ = self.gru(attn_output)
        
        # Take the last hidden state for classification
        x = gru_out[:, -1, :]
        
        # Dense Layers 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)
