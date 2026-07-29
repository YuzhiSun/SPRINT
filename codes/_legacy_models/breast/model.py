import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet, Bottleneck

class SmallResNet32(nn.Module):
    def __init__(self, block, layers):
        super(SmallResNet32, self).__init__()
        self.inplanes = 64
        
        # Input: 32x32
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # L1: 32x32 (s=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        # L2: 16x16 (s=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # L3: 8x8 (s=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # L4: 4x4 (s=2) -> 512 channels
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x # (B, 512, 4, 4)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None: identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ImageTower(nn.Module):
    def __init__(self):
        super(ImageTower, self).__init__()
        self.backbone = SmallResNet32(BasicBlock, [2, 2, 2, 2])
    def forward(self, x):
        return self.backbone(x)


class RNATower(nn.Module):
    def __init__(self, num_genes=None, output_dim=512): 
        super(RNATower, self).__init__()
        if num_genes is None:
            raise ValueError("num_genes must be provided")
        self.encoder = nn.Sequential(
            nn.Linear(num_genes, 2048), 
            nn.BatchNorm1d(2048), nn.ReLU(), nn.Dropout(0.4), 
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(1024, output_dim), 
            nn.BatchNorm1d(output_dim), nn.ReLU()
        )
    def forward(self, x):
        return self.encoder(x)


class RNAProcessor(nn.Module):
    def forward(self, x): return x.unsqueeze(1)


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model=512):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
    def forward(self, x):
        return x + self.pos_embed[:, :x.shape[1], :]


class ImageProcessor_A(nn.Module):
    def forward(self, x):
        # x: (B, 512, 4, 4) -> (B, 16, 512)
        b, c, h, w = x.shape
        return x.view(b, c, h*w).permute(0, 2, 1)


class ImageProcessor_B(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(512, 512, 1), nn.BatchNorm2d(512), nn.ReLU())
    def forward(self, x):
        x = self.proj(x)
        b, c, h, w = x.shape
        return x.view(b, c, h*w).permute(0, 2, 1)


class ImageProcessor_C(nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x):
        return self.avgpool(x).flatten(1).unsqueeze(1)


class BaseModel(nn.Module):
    def __init__(self, num_proteins, num_genes=None):
        super().__init__()
        self.img_tower = ImageTower()
        self.rna_tower = RNATower(num_genes=num_genes, output_dim=512)
        self.regressor = nn.Sequential(
            nn.Flatten(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, num_proteins)
        )


class Model_A1(BaseModel):
    def __init__(self, num_proteins, num_genes=None):
        super().__init__(num_proteins, num_genes=num_genes)
        self.img_proc = ImageProcessor_A()
        self.rna_proc = RNAProcessor()
        self.img_pos = LearnablePositionalEncoding(max_len=16)
        self.attn = nn.MultiheadAttention(512, 8, batch_first=True)
    def forward(self, img, rna):
        k = self.img_pos(self.img_proc(self.img_tower(img)))
        q = self.rna_proc(self.rna_tower(rna))
        out, _ = self.attn(q, k, k)
        return self.regressor(out + q)


class Model_A2(BaseModel):
    def __init__(self, num_proteins, num_genes=None):
        super().__init__(num_proteins, num_genes=num_genes)
        self.img_proc = ImageProcessor_A()
        self.rna_proc = RNAProcessor()
        self.pos_enc = LearnablePositionalEncoding(max_len=17)
        self.attn = nn.MultiheadAttention(512, 8, batch_first=True)
    def forward(self, img, rna):
        i = self.img_proc(self.img_tower(img))
        r = self.rna_proc(self.rna_tower(rna))
        x = self.pos_enc(torch.cat([i, r], dim=1))
        out, _ = self.attn(x, x, x)
        return self.regressor(out[:, -1:, :] + r)


class Model_B1(Model_A1):
    def __init__(self, num_proteins, num_genes=None):
        super().__init__(num_proteins, num_genes=num_genes)
        self.img_proc = ImageProcessor_B()


class Model_B2(Model_A2):
    def __init__(self, num_proteins, num_genes=None):
        super().__init__(num_proteins, num_genes=num_genes)
        self.img_proc = ImageProcessor_B()


class Model_C1(BaseModel):
    def __init__(self, num_proteins, num_genes=None):
        super().__init__(num_proteins, num_genes=num_genes)
        self.img_proc = ImageProcessor_C()
        self.rna_proc = RNAProcessor()
        self.attn = nn.MultiheadAttention(512, 8, batch_first=True)
    def forward(self, img, rna):
        k = self.img_proc(self.img_tower(img))
        q = self.rna_proc(self.rna_tower(rna))
        out, _ = self.attn(q, k, k)
        return self.regressor(out + q)


class Model_C2(BaseModel):
    def __init__(self, num_proteins, num_genes=None):
        super().__init__(num_proteins, num_genes=num_genes)
        self.img_proc = ImageProcessor_C()
        self.rna_proc = RNAProcessor()
        self.pos_enc = LearnablePositionalEncoding(max_len=2)
        self.attn = nn.MultiheadAttention(512, 8, batch_first=True)
    def forward(self, img, rna):
        i = self.img_proc(self.img_tower(img))
        r = self.rna_proc(self.rna_tower(rna))
        x = self.pos_enc(torch.cat([i, r], dim=1))
        out, _ = self.attn(x, x, x)
        return self.regressor(out[:, -1:, :] + r)


class Model_RNA_Only(nn.Module):
    def __init__(self, num_proteins, num_genes=None):
        super().__init__()
        self.rna_tower = RNATower(num_genes=num_genes, output_dim=512)
        self.regressor = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, num_proteins))
    def forward(self, img, rna):
        return self.regressor(self.rna_tower(rna))


class Model_A1_NoAttn(BaseModel):
    def __init__(self, num_proteins, num_genes=None):
        super().__init__(num_proteins, num_genes=num_genes)
        self.img_proc = ImageProcessor_A()
        self.rna_proc = RNAProcessor()
        self.fusion = nn.Linear(1024, 512)
    def forward(self, img, rna):
        i = self.img_proc(self.img_tower(img)).mean(1)
        r = self.rna_tower(rna)
        return self.regressor(torch.relu(self.fusion(torch.cat([i, r], 1))).unsqueeze(1))


class Model_C1_NoAttn(BaseModel):
    def __init__(self, num_proteins, num_genes=None):
        super().__init__(num_proteins, num_genes=num_genes)
        self.img_proc = ImageProcessor_C()
        self.rna_proc = RNAProcessor()
        self.fusion = nn.Linear(1024, 512)
    def forward(self, img, rna):
        i = self.img_proc(self.img_tower(img)).squeeze(1)
        r = self.rna_tower(rna)
        return self.regressor(torch.relu(self.fusion(torch.cat([i, r], 1))).unsqueeze(1))



