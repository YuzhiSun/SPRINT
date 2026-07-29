import torch
import torch.nn as nn
import torchvision.models as models

class ImageTower(nn.Module):
    def __init__(self, freeze=False):
        super().__init__()
        try:
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        except Exception:
            resnet = models.resnet50(weights=None)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        return self.backbone(x)


class RNATower(nn.Module):
    def __init__(self, num_genes, output_dim=2048):
        super().__init__()
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
            nn.ReLU(),
        )

    def forward(self, x):
        return self.encoder(x)


class ImageProcessorA(nn.Module):
    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).permute(0, 2, 1)
        x = x.view(b, h * w, 4, 512)
        return x.reshape(b, -1, 512)


class ImageProcessorC(nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.avgpool(x).flatten(1)
        b, _ = x.shape
        return x.view(b, 4, 512)


class RNAProcessor(nn.Module):
    def forward(self, x):
        b, _ = x.shape
        return x.view(b, 4, 512)


class PosEnc(nn.Module):
    def __init__(self, max_len):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, 512) * 0.02)

    def forward(self, x):
        return x + self.pos_embed[:, : x.shape[1], :]


class SelfAttentionMSIModel(nn.Module):
    def __init__(self, scheme, num_genes, num_metabolites, freeze_image=False):
        super().__init__()
        self.scheme = scheme
        self.img_tower = ImageTower(freeze=freeze_image)
        self.rna_tower = RNATower(num_genes=num_genes)
        self.rna_processor = RNAProcessor()
        if scheme == "A2":
            self.img_processor = ImageProcessorA()
            self.pos_enc = PosEnc(max_len=260)
        elif scheme == "C2":
            self.img_processor = ImageProcessorC()
            self.pos_enc = PosEnc(max_len=8)
        else:
            raise ValueError(scheme)
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


