import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        return F.relu(self.conv(x) + self.shortcut(x))


class SpatialFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, feature_dim=256):
        super().__init__()
        # Input: (B*T, in_channels, 112, 112)
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2, bias=False), # 56x56
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # 28x28
        )
        self.stage1 = ConvBlock(32, 64, stride=2)   # 14x14
        self.stage2 = ConvBlock(64, 128, stride=2)  # 7x7
        self.stage3 = ConvBlock(128, 256, stride=2) # 4x4
        self.pool = nn.AdaptiveAvgPool2d((1, 1))    # 1x1
        self.fc = nn.Linear(256, feature_dim)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (B, T, hidden_dim)
        scores = self.attn(x) # (B, T, 1)
        weights = F.softmax(scores, dim=1)
        context = torch.sum(x * weights, dim=1) # (B, hidden_dim)
        return context, weights


class ISLGestureRecognizer(nn.Module):
    def __init__(self, num_classes=5, num_frames=16, in_channels=3, feature_dim=256, hidden_dim=128):
        super().__init__()
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.in_channels = in_channels
        self.feature_dim = feature_dim
        
        # Spatial CNN
        self.spatial_extractor = SpatialFeatureExtractor(in_channels=in_channels, feature_dim=feature_dim)
        
        # Temporal Bi-directional GRU
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Temporal Attention
        self.attention = TemporalAttention(hidden_dim * 2)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x shape: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        
        # Extract spatial features
        spatial_feats = self.spatial_extractor(x_flat) # (B * T, feature_dim)
        spatial_feats = spatial_feats.view(B, T, self.feature_dim) # (B, T, feature_dim)
        
        # Temporal sequence processing
        gru_out, _ = self.gru(spatial_feats) # (B, T, hidden_dim * 2)
        
        # Attention aggregation
        context, attn_weights = self.attention(gru_out) # (B, hidden_dim * 2)
        
        # Final classification
        logits = self.classifier(context) # (B, num_classes)
        return logits, attn_weights
