import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet, Bottleneck

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ImageTower(nn.Module):
    def __init__(self, output_channels=2048):
        super(ImageTower, self).__init__()
        # 使用 ImageNet 预训练权重
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # ResNet50 的 layer4 输出本身就是 (Batch, 2048, 8, 8)
        self.backbone = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4
        )

    def forward(self, x):
        return self.backbone(x)


class RNATower(nn.Module):
    def __init__(self, num_genes=None, output_dim=2048):
        super(RNATower, self).__init__()
        
        if num_genes is None:
            raise ValueError("num_genes must be provided")
            
        # 输入维度变大，中间层也相应加宽
        self.encoder = nn.Sequential(
            nn.Linear(num_genes, 4096), 
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(0.4), 
            
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(2048, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.encoder(x)


class ImageProcessor_SchemeA(nn.Module):
    def __init__(self):
        super(ImageProcessor_SchemeA, self).__init__()
    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).permute(0, 2, 1) # (B, 64, 2048)
        x = x.view(b, h * w, 4, 512) # (B, 64, 4, 512)
        x = x.reshape(b, -1, 512) # (B, 256, 512)
        return x


class ImageProcessor_SchemeB(nn.Module):
    def __init__(self, input_dim=2048, target_dim=512):
        super(ImageProcessor_SchemeB, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(input_dim, target_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(target_dim),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.proj(x)
        b, c, h, w = x.shape
        return x.view(b, c, h * w).permute(0, 2, 1)


class ImageProcessor_SchemeC(nn.Module):
    def __init__(self):
        super(ImageProcessor_SchemeC, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x):
        x = self.avgpool(x).flatten(1)
        b, c = x.shape
        return x.view(b, 4, 512)


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model=512):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
    def forward(self, x):
        curr_len = x.shape[1]
        return x + self.pos_embed[:, :curr_len, :]


class RNAProcessor(nn.Module):
    def __init__(self):
        super(RNAProcessor, self).__init__()
    def forward(self, x):
        b, c = x.shape
        return x.view(b, 4, 512)

class Model_SchemeA1(nn.Module):
    def __init__(self, num_proteins, num_genes=None):
        super(Model_SchemeA1, self).__init__()
        self.img_tower = ImageTower()
        self.rna_tower = RNATower(num_genes=num_genes)
        self.img_processor = ImageProcessor_SchemeA()
        self.rna_processor = RNAProcessor()
        self.img_pos_enc = LearnablePositionalEncoding(max_len=256, d_model=512)
        self.cross_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        self.regressor = nn.Sequential(nn.Flatten(), nn.Linear(2048, 1024), nn.ReLU(), nn.Linear(1024, num_proteins))

    def forward(self, img, rna):
        img_feat = self.img_tower(img)
        rna_feat = self.rna_tower(rna)
        img_seq = self.img_pos_enc(self.img_processor(img_feat))
        rna_seq = self.rna_processor(rna_feat)
        attn_output, _ = self.cross_attn(query=rna_seq, key=img_seq, value=img_seq)
        return self.regressor(attn_output + rna_seq)


class Model_SchemeA2(nn.Module):
    def __init__(self, num_proteins, num_genes=None):
        super(Model_SchemeA2, self).__init__()
        self.img_tower = ImageTower()
        self.rna_tower = RNATower(num_genes=num_genes)
        self.img_processor = ImageProcessor_SchemeA()
        self.rna_processor = RNAProcessor()
        self.pos_enc = LearnablePositionalEncoding(max_len=260, d_model=512)
        self.self_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        self.regressor = nn.Sequential(nn.Flatten(), nn.Linear(2048, 1024), nn.ReLU(), nn.Linear(1024, num_proteins))

    def forward(self, img, rna):
        img_seq = self.img_processor(self.img_tower(img))
        rna_seq = self.rna_processor(self.rna_tower(rna))
        combined = self.pos_enc(torch.cat([img_seq, rna_seq], dim=1))
        attn_output, _ = self.self_attn(combined, combined, combined)
        return self.regressor(attn_output[:, -4:, :] + rna_seq)


class Model_SchemeB1(nn.Module):
    def __init__(self, num_proteins, num_genes=None):
        super(Model_SchemeB1, self).__init__()
        self.img_tower = ImageTower()
        self.rna_tower = RNATower(num_genes=num_genes)
        self.img_processor = ImageProcessor_SchemeB()
        self.rna_processor = RNAProcessor()
        self.img_pos_enc = LearnablePositionalEncoding(max_len=64, d_model=512)
        self.cross_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        self.regressor = nn.Sequential(nn.Flatten(), nn.Linear(2048, 1024), nn.ReLU(), nn.Linear(1024, num_proteins))

    def forward(self, img, rna):
        img_seq = self.img_pos_enc(self.img_processor(self.img_tower(img)))
        rna_seq = self.rna_processor(self.rna_tower(rna))
        attn_output, _ = self.cross_attn(query=rna_seq, key=img_seq, value=img_seq)
        return self.regressor(attn_output + rna_seq)


class Model_SchemeB2(nn.Module):
    def __init__(self, num_proteins, num_genes=None):
        super(Model_SchemeB2, self).__init__()
        self.img_tower = ImageTower()
        self.rna_tower = RNATower(num_genes=num_genes)
        self.img_processor = ImageProcessor_SchemeB()
        self.rna_processor = RNAProcessor()
        self.pos_enc = LearnablePositionalEncoding(max_len=68, d_model=512)
        self.self_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        self.regressor = nn.Sequential(nn.Flatten(), nn.Linear(2048, 1024), nn.ReLU(), nn.Linear(1024, num_proteins))

    def forward(self, img, rna):
        img_seq = self.img_processor(self.img_tower(img))
        rna_seq = self.rna_processor(self.rna_tower(rna))
        combined = self.pos_enc(torch.cat([img_seq, rna_seq], dim=1))
        attn_output, _ = self.self_attn(combined, combined, combined)
        return self.regressor(attn_output[:, -4:, :] + rna_seq)


class Model_SchemeC1(nn.Module):
    def __init__(self, num_proteins, num_genes=None):
        super(Model_SchemeC1, self).__init__()
        self.img_tower = ImageTower()
        self.rna_tower = RNATower(num_genes=num_genes)
        self.img_processor = ImageProcessor_SchemeC()
        self.rna_processor = RNAProcessor()
        self.img_pos_enc = LearnablePositionalEncoding(max_len=4, d_model=512)
        self.cross_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        self.regressor = nn.Sequential(nn.Flatten(), nn.Linear(2048, 1024), nn.ReLU(), nn.Linear(1024, num_proteins))

    def forward(self, img, rna):
        img_seq = self.img_pos_enc(self.img_processor(self.img_tower(img)))
        rna_seq = self.rna_processor(self.rna_tower(rna))
        attn_output, _ = self.cross_attn(query=rna_seq, key=img_seq, value=img_seq)
        return self.regressor(attn_output + rna_seq)


class Model_SchemeC2(nn.Module):
    def __init__(self, num_proteins, num_genes=None):
        super(Model_SchemeC2, self).__init__()
        self.img_tower = ImageTower()
        self.rna_tower = RNATower(num_genes=num_genes)
        self.img_processor = ImageProcessor_SchemeC()
        self.rna_processor = RNAProcessor()
        self.pos_enc = LearnablePositionalEncoding(max_len=8, d_model=512)
        self.self_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        self.regressor = nn.Sequential(nn.Flatten(), nn.Linear(2048, 1024), nn.ReLU(), nn.Linear(1024, num_proteins))

    def forward(self, img, rna):
        img_seq = self.img_processor(self.img_tower(img))
        rna_seq = self.rna_processor(self.rna_tower(rna))
        combined = self.pos_enc(torch.cat([img_seq, rna_seq], dim=1))
        attn_output, _ = self.self_attn(combined, combined, combined)
        return self.regressor(attn_output[:, -4:, :] + rna_seq)


class Model_RNA_Only(nn.Module):
    def __init__(self, num_proteins, num_genes=None):
        super(Model_RNA_Only, self).__init__()
        self.rna_tower = RNATower(num_genes=num_genes)
        self.rna_processor = RNAProcessor()
        self.regressor = nn.Sequential(nn.Flatten(), nn.Linear(2048, 1024), nn.ReLU(), nn.Linear(1024, num_proteins))
    def forward(self, img, rna):
        rna_seq = self.rna_processor(self.rna_tower(rna))
        return self.regressor(rna_seq)


class Model_SchemeA2_HE_Only(nn.Module):
    """Ablation: Scheme A2 with RNA branch removed (H&E image only)"""
    def __init__(self, num_proteins, num_genes=None):
        super().__init__()
        self.img_tower = ImageTower()
        self.img_processor = ImageProcessor_SchemeA()
        self.pos_enc = LearnablePositionalEncoding(max_len=256, d_model=512)
        self.self_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        self.regressor = nn.Sequential(
            nn.Flatten(), nn.Linear(2048, 1024), nn.ReLU(), nn.Linear(1024, num_proteins)
        )

    def forward(self, img, rna):
        img_seq = self.img_processor(self.img_tower(img))           # (B, 256, 512)
        img_seq = self.pos_enc(img_seq)
        attn_output, _ = self.self_attn(img_seq, img_seq, img_seq) # (B, 256, 512)
        pooled = attn_output.mean(dim=1, keepdim=True).expand(-1, 4, -1)  # (B, 4, 512)
        return self.regressor(pooled)


class Model_SchemeA1_NoAttn(nn.Module):
    def __init__(self, num_proteins, num_genes=None):
        super(Model_SchemeA1_NoAttn, self).__init__()
        self.img_tower = ImageTower()
        self.rna_tower = RNATower(num_genes=num_genes)
        self.img_processor = ImageProcessor_SchemeA()
        self.rna_processor = RNAProcessor()
        self.fusion_layer = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.regressor = nn.Sequential(nn.Flatten(), nn.Linear(2048, 1024), nn.ReLU(), nn.Linear(1024, num_proteins))
    def forward(self, img, rna):
        img_mean = torch.mean(self.img_processor(self.img_tower(img)), dim=1, keepdim=True).expand(-1, 4, -1)
        rna_seq = self.rna_processor(self.rna_tower(rna))
        combined = torch.cat([rna_seq, img_mean], dim=-1)
        return self.regressor(self.fusion_layer(combined))


class Model_SchemeC1_NoAttn(nn.Module):
    def __init__(self, num_proteins, num_genes=None):
        super(Model_SchemeC1_NoAttn, self).__init__()
        self.img_tower = ImageTower()
        self.rna_tower = RNATower(num_genes=num_genes)
        self.img_processor = ImageProcessor_SchemeC()
        self.rna_processor = RNAProcessor()
        # Input to fusion: 512(RNA) + 512(Img) = 1024
        self.fusion_layer = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.regressor = nn.Sequential(nn.Flatten(), nn.Linear(2048, 1024), nn.ReLU(), nn.Linear(1024, num_proteins))
    
    def forward(self, img, rna):
        # img_seq: (Batch, 4, 512) for Scheme C
        img_seq = self.img_processor(self.img_tower(img))
        rna_seq = self.rna_processor(self.rna_tower(rna))
        # Simple concat fusion without Attention mechanism
        combined = torch.cat([rna_seq, img_seq], dim=-1) # (Batch, 4, 1024)
        fused = self.fusion_layer(combined) # (Batch, 4, 512)
        return self.regressor(fused) 
