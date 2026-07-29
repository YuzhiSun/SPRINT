import torch
import torch.nn as nn
import torchvision.models as models

# ==========================================
# 1. 基础组件: ResNet Block (保持不变)
# ==========================================
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

# ==========================================
# 2. 重构版 ImageTower (ResNet50 NoPool -> 2048ch)
# ==========================================
class ImageTower(nn.Module):
    def __init__(self, output_channels=2048):
        super(ImageTower, self).__init__()
        # 使用 torchvision 的 ResNet50 结构
        
        # [原始方案] weights=None 表示从头训练 (Random Initialization)
        # self.resnet = models.resnet50(weights=None)
        
        # [优化方案] 使用 ImageNet 预训练权重
        #print("Using ImageNet Pre-trained Weights for ImageTower")
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # 移除最后两层 (AvgPool 和 FC)
        # ResNet50 的 layer4 输出本身就是 (Batch, 2048, 8, 8)
        # 所以我们只需要保留到 layer4 即可
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
        # x: (Batch, 3, 256, 256)
        out = self.backbone(x)
        # ResNet50 layer4 output: (Batch, 2048, 8, 8)
        return out

# ==========================================
# 3. 重构版 RNATower (Full Genes -> 2048ch)
# ==========================================
class RNATower(nn.Module):
    def __init__(self, num_genes=18085, output_dim=2048):
        super(RNATower, self).__init__()
        
        # 输入维度变大 (18085)，中间层也相应加宽
        self.encoder = nn.Sequential(
            nn.Linear(num_genes, 4096), # 第一层放大或保持
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(0.4), # 防止过拟合
            
            nn.Linear(4096, 2048), # 压缩到目标维度
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # 最后一层保持 2048，不加激活函数，或者加一个 Linear 整理
            nn.Linear(2048, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        # x: (Batch, 18085)
        return self.encoder(x) # -> (Batch, 2048)

# ==========================================
# 方案 A: 空间展平 + 通道分组 (Spatial Flatten + Channel Grouping)
# ==========================================
class ImageProcessor_SchemeA(nn.Module):
    def __init__(self):
        super(ImageProcessor_SchemeA, self).__init__()
        # 方案 A 不需要额外的参数层，纯粹是维度变换
        
    def forward(self, x):
        # 输入 x: (Batch, 2048, 8, 8) 来自 ResNet50 Layer4
        
        # 1. 空间展平: (Batch, 2048, 8, 8) -> (Batch, 2048, 64)
        b, c, h, w = x.shape
        x = x.view(b, c, h * w)
        
        # 2. 维度置换: (Batch, 2048, 64) -> (Batch, 64, 2048)
        x = x.permute(0, 2, 1)
        
        # 3. 通道拆分 (Reshape): (Batch, 64, 2048) -> (Batch, 64, 4, 512)
        # 将 2048 拆分为 4 个 512
        x = x.view(b, h * w, 4, 512)
        
        # 4. 序列合并: (Batch, 64, 4, 512) -> (Batch, 256, 512)
        # 将 "空间位置(64)" 和 "通道分组(4)" 合并为一个长序列
        x = x.reshape(b, -1, 512)
        
        return x # 输出: (Batch, 256, 512)


# ==========================================
# 方案 C: 全局池化 + 通道拆分 (Global Pooling + Channel Splitting)
# ==========================================
class ImageProcessor_SchemeC(nn.Module):
    def __init__(self):
        super(ImageProcessor_SchemeC, self).__init__()
        # 定义全局平均池化层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        # 输入 x: (Batch, 2048, 8, 8) 来自 ResNet50 Layer4
        
        # 1. 全局平均池化: (Batch, 2048, 8, 8) -> (Batch, 2048, 1, 1)
        x = self.avgpool(x)
        
        # 2. 展平: (Batch, 2048, 1, 1) -> (Batch, 2048)
        x = torch.flatten(x, 1)
        
        # 3. 通道拆分 (Reshape): (Batch, 2048) -> (Batch, 4, 512)
        # 将 2048 维向量拆分为 4 个 512 维的 Token
        b, c = x.shape
        x = x.view(b, 4, 512)
        
        return x # 输出: (Batch, 4, 512)



# ==========================================
# 0. 通用组件: 位置编码 (Positional Encoding)
# ==========================================
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model=512):
        super().__init__()
        # 初始化为很小的随机数
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x):
        # x: (Batch, Seq_Len, Dim)
        # 自动广播: (1, Max_Len, Dim) -> (Batch, Seq_Len, Dim)
        # 截取对应长度的位置编码 (以防输入长度小于 max_len)
        curr_len = x.shape[1]
        return x + self.pos_embed[:, :curr_len, :]

# ==========================================
# 辅助模块: RNA Processor
# ==========================================
class RNAProcessor(nn.Module):
    def __init__(self):
        super(RNAProcessor, self).__init__()
        
    def forward(self, x):
        # x: (Batch, 2048)
        b, c = x.shape
        return x.view(b, 4, 512)


# ==========================================
# 方案 A2: Self Attention (Image Scheme A)
# Image: Spatial Flatten (256 tokens)
# ==========================================
class Model_SchemeA2(nn.Module):
    def __init__(self, num_proteins, num_genes=18085):
        super(Model_SchemeA2, self).__init__()
        self.img_tower = ImageTower()
        self.rna_tower = RNATower(num_genes=num_genes)
        self.img_processor = ImageProcessor_SchemeA()
        self.rna_processor = RNAProcessor()
        
        # Positional Encoding for Combined Sequence (256 + 4 = 260)
        # 0-255: Image, 256-259: RNA
        self.pos_enc = LearnablePositionalEncoding(max_len=260, d_model=512)
        
        self.self_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_proteins)
        )

    def forward(self, img, rna):
        img_feat = self.img_tower(img)
        rna_feat = self.rna_tower(rna)
        img_seq = self.img_processor(img_feat) # (B, 256, 512)
        rna_seq = self.rna_processor(rna_feat) # (B, 4, 512)
        
        # Concat
        combined_seq = torch.cat([img_seq, rna_seq], dim=1) 
        
        # Add PE (Crucial for distinguishing Image vs RNA and Spatial locations)
        combined_seq = self.pos_enc(combined_seq)
        
        attn_output, _ = self.self_attn(combined_seq, combined_seq, combined_seq)
        
        rna_out = attn_output[:, -4:, :] + rna_seq
        return self.regressor(rna_out)

# ==========================================
# 方案 C2: Self Attention (Image Scheme C)
# Image: Global Pool (4 tokens)
# ==========================================
class Model_SchemeC2(nn.Module):
    def __init__(self, num_proteins, num_genes=18085):
        super(Model_SchemeC2, self).__init__()
        self.img_tower = ImageTower()
        self.rna_tower = RNATower(num_genes=num_genes)
        self.img_processor = ImageProcessor_SchemeC()
        self.rna_processor = RNAProcessor()
        
        # PE for Combined (4 + 4 = 8)
        self.pos_enc = LearnablePositionalEncoding(max_len=8, d_model=512)
        
        self.self_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_proteins)
        )

    def forward(self, img, rna):
        img_feat = self.img_tower(img)
        rna_feat = self.rna_tower(rna)
        img_seq = self.img_processor(img_feat) # (B, 4, 512)
        rna_seq = self.rna_processor(rna_feat) # (B, 4, 512)
        
        combined_seq = torch.cat([img_seq, rna_seq], dim=1) 
        
        # Add PE
        combined_seq = self.pos_enc(combined_seq)
        
        attn_output, _ = self.self_attn(combined_seq, combined_seq, combined_seq)
        
        rna_out = attn_output[:, -4:, :] + rna_seq
        return self.regressor(rna_out)

# ==========================================
# 8. 消融实验 (Ablation Studies) - 优化版
# ==========================================
# 核心原则: 
# 1. 保持参数量级大致相当 (用 Linear 替代 Attention)
# 2. 使用 Concat 而不是 Add (避免特征空间未对齐导致的干扰)

# --- 1. 仅 RNA 模型 (Baseline: Remove Image) ---
class Model_RNA_Only(nn.Module):
    def __init__(self, num_proteins, num_genes=18085):
        super(Model_RNA_Only, self).__init__()
        # 保持和主模型一样的提取器，确保公平
        self.rna_tower = RNATower(num_genes=num_genes)
        self.rna_processor = RNAProcessor()
        
        # Regressor
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_proteins)
        )

    def forward(self, img, rna):
        # 完全忽略 img 输入
        rna_feat = self.rna_tower(rna)
        rna_seq = self.rna_processor(rna_feat) # (B, 4, 512)
        return self.regressor(rna_seq)

# --- 2. A1 去除 Attention (Spatial Mean + Concat) ---
# 验证点: 图像的"细粒度空间分布"是否重要？
# 做法: 将图像特征强行平均化 (Global Average Pooling)，丢失空间信息，然后与 RNA 拼接
class Model_SchemeA1_NoAttn(nn.Module):
    def __init__(self, num_proteins, num_genes=18085):
        super(Model_SchemeA1_NoAttn, self).__init__()
        self.img_tower = ImageTower()
        self.rna_tower = RNATower(num_genes=num_genes)
        self.img_processor = ImageProcessor_SchemeA()
        self.rna_processor = RNAProcessor()
        
        # 替代 Attention 的融合层
        # 输入: RNA(512) + Image(512) = 1024
        # 输出: 512 (恢复到和 Attention 输出一样的维度)
        self.fusion_layer = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU()
        )
        
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_proteins)
        )

    def forward(self, img, rna):
        img_feat = self.img_tower(img)
        rna_feat = self.rna_tower(rna)
        
        img_seq = self.img_processor(img_feat) # (B, 256, 512)
        rna_seq = self.rna_processor(rna_feat) # (B, 4, 512)
        
        # 1. 破坏空间结构: 取平均
        # (B, 256, 512) -> (B, 1, 512)
        img_mean = torch.mean(img_seq, dim=1, keepdim=True)
        
        # 2. 扩展 Image 维度以匹配 RNA 序列长度
        # (B, 1, 512) -> (B, 4, 512)
        img_mean_expanded = img_mean.expand(-1, 4, -1)
        
        # 3. 拼接 (Concat) 代替 Attention
        # (B, 4, 512+512) -> (B, 4, 1024)
        combined = torch.cat([rna_seq, img_mean_expanded], dim=-1)
        
        # 4. 融合映射
        # (B, 4, 1024) -> (B, 4, 512)
        x = self.fusion_layer(combined)
        
        return self.regressor(x)

# --- 3. C1 去除 Attention (Simple Concat) ---
# 验证点: "Attention 交互机制" 是否比 "简单全连接融合" 更好？
# 做法: 直接拼接两个模态的特征，不进行 Query-Key 匹配
class Model_SchemeC1_NoAttn(nn.Module):
    def __init__(self, num_proteins, num_genes=18085):
        super(Model_SchemeC1_NoAttn, self).__init__()
        self.img_tower = ImageTower()
        self.rna_tower = RNATower(num_genes=num_genes)
        self.img_processor = ImageProcessor_SchemeC()
        self.rna_processor = RNAProcessor()
        
        # 替代 Attention 的融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU()
        )
        
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_proteins)
        )

    def forward(self, img, rna):
        img_feat = self.img_tower(img)
        rna_feat = self.rna_tower(rna)
        
        img_seq = self.img_processor(img_feat) # (B, 4, 512)
        rna_seq = self.rna_processor(rna_feat) # (B, 4, 512)
        
        # 1. 直接拼接 (Concat)
        # 假设 C1 也是序列长度为 4，直接对应位置拼接
        combined = torch.cat([rna_seq, img_seq], dim=-1) # (B, 4, 1024)
        
        # 2. 融合映射
        x = self.fusion_layer(combined) # (B, 4, 512)
        
        return self.regressor(x)


