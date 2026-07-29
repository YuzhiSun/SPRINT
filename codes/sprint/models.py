"""
SPRINT model architectures — Spatial Protein Inference from Transcriptomics.

Architecture family
-------------------
- SPRINT_A2  (Scheme A2):  ResNet50 backbone, 256×256 H&E crops  → Brain, Tonsil
- SPRINT_C2  (Scheme C2):  SmallResNet backbone, 14-32 px crops   → Spleen, Breast
- SPRINT_MSI (Scheme A2):  ResNet50 backbone, MSI metabolomics   → MSI Mouse Brain

Ablation variants
-----------------
- SPRINT_RNAOnly_A / SPRINT_RNAOnly_C : RNA-only baselines
- SPRINT_HE_Only                      : H&E-only baseline
- SPRINT_NoAttn_A1 / SPRINT_NoAttn_C1 : No-attention fusion baselines
"""

import torch
import torch.nn as nn
import torchvision.models as models

# ===========================================================================
# Shared building blocks
# ===========================================================================

class BasicBlock(nn.Module):
    """Standard residual block used by SmallResNet backbones."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.relu(out)


class LearnablePositionalEncoding(nn.Module):
    """Learned position embeddings added to token sequence."""

    def __init__(self, max_len, d_model=512):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x):
        return x + self.pos_embed[:, : x.shape[1], :]


# ===========================================================================
# Image Towers
# ===========================================================================

class ResNet50Tower(nn.Module):
    """ResNet50 backbone (ImageNet pretrained), output 2048×H×W feature map.

    Used by: SPRINT_A2 (Brain, Tonsil), SPRINT_MSI
    """

    def __init__(self, freeze=False):
        super().__init__()
        try:
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        except Exception:
            resnet = models.resnet50(weights=None)
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        return self.backbone(x)  # (B, 2048, 8, 8)


class SmallResNetTower(nn.Module):
    """Small ResNet-18-depth backbone for compact H&E crops.

    Parameters
    ----------
    image_size : int
        32 for Breast, 14 for Spleen. Controls layer4 stride.
    """

    def __init__(self, image_size=32):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        # layer4 stride: 32→stride2, 14→stride1 (both yield 4×4 spatial)
        layer4_stride = 1 if image_size <= 14 else 2
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=layer4_stride)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride)]
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
        return x  # (B, 512, 4, 4)


# ===========================================================================
# RNA Towers
# ===========================================================================

class RNATowerA(nn.Module):
    """Large MLP RNA encoder for Scheme A (output 2048-d)."""

    def __init__(self, num_genes, output_dim=2048):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_genes, 4096),
            nn.BatchNorm1d(4096), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(2048, output_dim),
            nn.BatchNorm1d(output_dim), nn.ReLU(),
        )

    def forward(self, x):
        return self.encoder(x)


class RNATowerC(nn.Module):
    """Compact MLP RNA encoder for Scheme C (output 512-d)."""

    def __init__(self, num_genes, output_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_genes, 2048),
            nn.BatchNorm1d(2048), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(1024, output_dim),
            nn.BatchNorm1d(output_dim), nn.ReLU(),
        )

    def forward(self, x):
        return self.encoder(x)


# ===========================================================================
# Image Processors
# ===========================================================================

class ImageProcessorA(nn.Module):
    """Scheme A: spatial flatten + channel grouping.

    Input:  (B, 2048, 8, 8)  [ResNet50 layer4]
    Output: (B, 256, 512)     [64 spatial × 4 channel groups]
    """

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).permute(0, 2, 1)   # (B, 64, 2048)
        x = x.view(b, h * w, 4, 512)                 # (B, 64, 4, 512)
        return x.reshape(b, -1, 512)                  # (B, 256, 512)


class ImageProcessorC(nn.Module):
    """Scheme C: global average pooling → single token.

    Input:  (B, 512, H', W')  [SmallResNet output]
    Output: (B, 1, 512)
    """

    def __init__(self):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.avgpool(x).flatten(1)   # (B, 512)
        return x.unsqueeze(1)             # (B, 1, 512)


# ===========================================================================
# RNA Processors
# ===========================================================================

class RNAProcessorA(nn.Module):
    """Split 2048-d RNA latent into 4 tokens of 512-d each."""

    def forward(self, x):
        b, c = x.shape
        return x.view(b, 4, 512)


class RNAProcessorC(nn.Module):
    """Wrap 512-d RNA latent as a single token."""

    def forward(self, x):
        return x.unsqueeze(1)  # (B, 1, 512)


# ===========================================================================
# SPRINT Model Family
# ===========================================================================

class SPRINT_A2(nn.Module):
    """SPRINT Scheme A2 — ResNet50 backbone for large H&E crops (256×256 px).

    Architecture
    ------------
    - H&E → ResNet50 (ImageNet pretrained) → (B,2048,8,8) feature map
    - RNA → MLP tower (N→4096→2048→2048)
    - Image processor: spatial flatten → 256 tokens × 512-d
    - RNA processor:   split 2048-d → 4 tokens × 512-d
    - Positional encoding on concatenated 260-token sequence
    - Self-attention (Multihead, 8 heads)
    - Readout from RNA residual tokens → regressor

    Used for: Brain (256px), Tonsil (256px)
    """

    def __init__(self, num_proteins, num_genes=None):
        super().__init__()
        self.img_tower = ResNet50Tower()
        self.rna_tower = RNATowerA(num_genes=num_genes, output_dim=2048)
        self.img_processor = ImageProcessorA()
        self.rna_processor = RNAProcessorA()
        self.pos_enc = LearnablePositionalEncoding(max_len=260, d_model=512)
        self.self_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_proteins),
        )

    def forward(self, img, rna):
        img_feat = self.img_tower(img)
        rna_feat = self.rna_tower(rna)
        img_seq = self.img_processor(img_feat)          # (B, 256, 512)
        rna_seq = self.rna_processor(rna_feat)          # (B, 4, 512)
        combined = self.pos_enc(torch.cat([img_seq, rna_seq], dim=1))
        attn_out, _ = self.self_attn(combined, combined, combined)
        rna_out = attn_out[:, -4:, :] + rna_seq          # RNA residual
        return self.regressor(rna_out)


class SPRINT_C2(nn.Module):
    """SPRINT Scheme C2 — SmallResNet backbone for compact H&E crops.

    Architecture
    ------------
    - H&E → SmallResNet → (B,512,H',W') feature map
    - RNA → Compact MLP (N→2048→1024→512)
    - Image processor: global pool → 1 token × 512-d
    - RNA processor:   wrap 512-d → 1 token × 512-d
    - Positional encoding on concatenated 2-token sequence
    - Self-attention (Multihead, 8 heads)
    - Readout from RNA residual → regressor

    Parameters
    ----------
    image_size : int
        32 for Breast, 14 for Spleen. Controls SmallResNet stride pattern.

    Used for: Breast (32px), Spleen (14px)
    """

    def __init__(self, num_proteins, num_genes=None, image_size=32):
        super().__init__()
        self.img_tower = SmallResNetTower(image_size=image_size)
        self.rna_tower = RNATowerC(num_genes=num_genes, output_dim=512)
        self.img_proc = ImageProcessorC()
        self.rna_proc = RNAProcessorC()
        self.pos_enc = LearnablePositionalEncoding(max_len=2)
        self.attn = nn.MultiheadAttention(512, 8, batch_first=True)
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_proteins),
        )

    def forward(self, img, rna):
        i = self.img_proc(self.img_tower(img))           # (B, 1, 512)
        r = self.rna_proc(self.rna_tower(rna))           # (B, 1, 512)
        x = self.pos_enc(torch.cat([i, r], dim=1))
        out, _ = self.attn(x, x, x)
        return self.regressor(out[:, -1:, :] + r)         # RNA residual


class SPRINT_MSI(nn.Module):
    """SPRINT for MSI metabolomics — ResNet50 + scheme-switchable processor.

    Parameters
    ----------
    scheme : str
        "A2" or "C2".
    freeze_image : bool
        If True, freeze ResNet50 backbone.
    """

    def __init__(self, scheme, num_genes, num_metabolites, freeze_image=False):
        super().__init__()
        self.scheme = scheme
        self.img_tower = ResNet50Tower(freeze=freeze_image)
        self.rna_tower = RNATowerA(num_genes=num_genes, output_dim=2048)
        self.rna_processor = RNAProcessorA()
        if scheme == "A2":
            self.img_processor = ImageProcessorA()
            self.pos_enc = LearnablePositionalEncoding(max_len=260)
        elif scheme == "C2":
            self.img_processor = ImageProcessorC()
            self.pos_enc = LearnablePositionalEncoding(max_len=8)
        else:
            raise ValueError(f"Unknown scheme: {scheme}")
        self.self_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_metabolites),
        )

    def forward(self, img, rna):
        img_seq = self.img_processor(self.img_tower(img))
        rna_seq = self.rna_processor(self.rna_tower(rna))
        combined = self.pos_enc(torch.cat([img_seq, rna_seq], dim=1))
        out, _ = self.self_attn(combined, combined, combined)
        return self.regressor(out[:, -4:, :] + rna_seq)


# ===========================================================================
# Ablation Models
# ===========================================================================

class SPRINT_RNAOnly_A(nn.Module):
    """RNA-only ablation (Scheme A scale). Same RNA tower as SPRINT_A2."""

    def __init__(self, num_proteins, num_genes=None):
        super().__init__()
        self.rna_tower = RNATowerA(num_genes=num_genes, output_dim=2048)
        self.rna_processor = RNAProcessorA()
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_proteins),
        )

    def forward(self, img, rna):
        return self.regressor(self.rna_processor(self.rna_tower(rna)))


class SPRINT_RNAOnly_C(nn.Module):
    """RNA-only ablation (Scheme C scale). Same RNA tower as SPRINT_C2."""

    def __init__(self, num_proteins, num_genes=None):
        super().__init__()
        self.rna_tower = RNATowerC(num_genes=num_genes, output_dim=512)
        self.regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_proteins),
        )

    def forward(self, img, rna):
        return self.regressor(self.rna_tower(rna))


class SPRINT_HE_Only(nn.Module):
    """H&E-only ablation (Scheme A2 with RNA branch removed)."""

    def __init__(self, num_proteins, num_genes=None):
        super().__init__()
        self.img_tower = ResNet50Tower()
        self.img_processor = ImageProcessorA()
        self.pos_enc = LearnablePositionalEncoding(max_len=256, d_model=512)
        self.self_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        self.regressor = nn.Sequential(
            nn.Flatten(), nn.Linear(2048, 1024), nn.ReLU(), nn.Linear(1024, num_proteins)
        )

    def forward(self, img, rna):
        img_seq = self.img_processor(self.img_tower(img))          # (B, 256, 512)
        img_seq = self.pos_enc(img_seq)
        attn_output, _ = self.self_attn(img_seq, img_seq, img_seq) # (B, 256, 512)
        pooled = attn_output.mean(dim=1, keepdim=True).expand(-1, 4, -1)  # (B, 4, 512)
        return self.regressor(pooled)


class SPRINT_NoAttn_A1(nn.Module):
    """Scheme A without attention: spatial mean + concat fusion."""

    def __init__(self, num_proteins, num_genes=None):
        super().__init__()
        self.img_tower = ResNet50Tower()
        self.rna_tower = RNATowerA(num_genes=num_genes, output_dim=2048)
        self.img_processor = ImageProcessorA()
        self.rna_processor = RNAProcessorA()
        self.fusion_layer = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.regressor = nn.Sequential(
            nn.Flatten(), nn.Linear(2048, 1024), nn.ReLU(), nn.Linear(1024, num_proteins)
        )

    def forward(self, img, rna):
        img_mean = torch.mean(self.img_processor(self.img_tower(img)), dim=1, keepdim=True)
        img_mean = img_mean.expand(-1, 4, -1)                     # (B, 4, 512)
        rna_seq = self.rna_processor(self.rna_tower(rna))         # (B, 4, 512)
        combined = torch.cat([rna_seq, img_mean], dim=-1)         # (B, 4, 1024)
        return self.regressor(self.fusion_layer(combined))


class SPRINT_NoAttn_C1(nn.Module):
    """Scheme C without attention: simple concat fusion."""

    def __init__(self, num_proteins, num_genes=None):
        super().__init__()
        self.img_tower = ResNet50Tower()
        self.rna_tower = RNATowerA(num_genes=num_genes, output_dim=2048)
        self.img_processor = ImageProcessorC()
        self.rna_processor = RNAProcessorA()
        self.fusion_layer = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.regressor = nn.Sequential(
            nn.Flatten(), nn.Linear(2048, 1024), nn.ReLU(), nn.Linear(1024, num_proteins)
        )

    def forward(self, img, rna):
        img_seq = self.img_processor(self.img_tower(img))
        rna_seq = self.rna_processor(self.rna_tower(rna))
        combined = torch.cat([rna_seq, img_seq], dim=-1)
        return self.regressor(self.fusion_layer(combined))


# ===========================================================================
# Legacy aliases — for backward compatibility with notebook model registries
# ===========================================================================

# Brain / Tonsil notebooks use these names:
Model_SchemeA2 = SPRINT_A2
Model_RNA_Only  = SPRINT_RNAOnly_A

# Spleen notebook uses these names:
Model_C2 = SPRINT_C2

# Tonsil notebook also uses:
Model_SchemeC2         = SPRINT_C2
Model_SchemeA2_HE_Only = SPRINT_HE_Only

# MSI notebook uses:
SelfAttentionMSIModel = SPRINT_MSI
