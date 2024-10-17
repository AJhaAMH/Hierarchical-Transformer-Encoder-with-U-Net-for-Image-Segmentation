import torch
import torch.nn as nn
from timm import create_model


class PatchMerging(nn.Module):

    def __init__(
            self,
            dim
    ):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4*dim, 2*dim, bias=False)

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
        x = self.norm(x)
        x = self.reduction(x)
        return x

class PatchExpansion(nn.Module):

    def __init__(
            self,
            dim
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim//2)
        self.expand = nn.Linear(dim, 2*dim, bias=False)

    def forward(self, x):

        x = self.expand(x)
        B, H, W, C = x.shape

        x = x.view(B, H , W, 2, 2, C//4)
        x = x.permute(0,1,3,2,4,5)

        x = x.reshape(B,H*2, W*2 , C//4)

        x = self.norm(x)
        return x
class SwinBlock(nn.Module):
  def __init__(self, dims, ip_res, num_heads=4, window_size=7, shift_size=3, mlp_ratio=4.):
        super().__init__()
        
        # Example of Swin Transformer layer structure
        self.layer_norm1 = nn.LayerNorm(dims)
        self.attn = nn.MultiheadAttention(embed_dim=dims, num_heads=num_heads)
        
        # Window-based MSA (Multi-head Self Attention)
        self.window_size = window_size
        self.shift_size = shift_size
        
        self.mlp = nn.Sequential(
            nn.Linear(dims, int(dims * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dims * mlp_ratio), dims)
        )
        self.layer_norm2 = nn.LayerNorm(dims)

  def forward(self, x):
        # Apply layer normalization and attention
        x = self.layer_norm1(x)
        attn_output, _ = self.attn(x, x, x)
        x = x + attn_output

        # Apply MLP and residual connection
        x = x + self.mlp(self.layer_norm2(x))
        
        return x

class Encoder(nn.Module):
    def __init__(self, C, partioned_ip_res, num_blocks=3):
        super().__init__()
        H,W = partioned_ip_res[0], partioned_ip_res[1]
        self.enc_swin_blocks = nn.ModuleList([
            SwinBlock(C, (H, W)),
            SwinBlock(2*C, (H//2, W//2)),
            SwinBlock(4*C, (H//4, W//4))
        ])
        self.enc_patch_merge_blocks = nn.ModuleList([
            PatchMerging(C),
            PatchMerging(2*C),
            PatchMerging(4*C)
        ])

    def forward(self, x):
        skip_conn_ftrs = []
        for swin_block,patch_merger in zip(self.enc_swin_blocks, self.enc_patch_merge_blocks):
            x = swin_block(x)
            skip_conn_ftrs.append(x)
            x = patch_merger(x)
        return x, skip_conn_ftrs


class Decoder(nn.Module):
    def __init__(self, C, partioned_ip_res, num_blocks=3):
        super().__init__()
        H,W = partioned_ip_res[0], partioned_ip_res[1]
        self.dec_swin_blocks = nn.ModuleList([
            SwinBlock(4*C, (H//4, W//4)),
            SwinBlock(2*C, (H//2, W//2)),
            SwinBlock(C, (H, W))
        ])
        self.dec_patch_expand_blocks = nn.ModuleList([
            PatchExpansion(8*C),
            PatchExpansion(4*C),
            PatchExpansion(2*C)
        ])
        self.skip_conn_concat = nn.ModuleList([
            nn.Linear(8*C, 4*C),
            nn.Linear(4*C, 2*C),
            nn.Linear(2*C, 1*C)
        ])

    def forward(self, x, encoder_features):
        for patch_expand,swin_block, enc_ftr, linear_concatter in zip(self.dec_patch_expand_blocks, self.dec_swin_blocks, encoder_features,self.skip_conn_concat):
            x = patch_expand(x)
            x = torch.cat([x, enc_ftr], dim=-1)
            x = linear_concatter(x)
            x = swin_block(x)
        return x


class SwinUNet(nn.Module):
    def __init__(self, ch, C, num_class, num_blocks=3, patch_size=4):
        super(SwinUNet, self).__init__()
        self.patch_size = patch_size
        self.patch_embed = PatchMerging(C)
        self.encoder = Encoder(C, (patch_size, patch_size), num_blocks)
        self.bottleneck = SwinBlock(C * (2**num_blocks), (patch_size // (2**num_blocks), patch_size // (2**num_blocks)))
        self.decoder = Decoder(C, (patch_size, patch_size), num_blocks)
        self.final_expansion = PatchExpansion(C)
        self.head = nn.Conv2d(C, num_class, 1, padding='same')

    def forward(self, x):
        # Infer height (H) and width (W) from input tensor
        H, W = x.shape[2], x.shape[3]  # Batch, Channels, Height, Width

        # Adjust encoder, bottleneck, and decoder dimensions dynamically
        self.encoder.input_shape = (H // self.patch_size, W // self.patch_size)
        self.bottleneck.input_shape = (H // (self.patch_size * (2**self.encoder.num_blocks)),
                                       W // (self.patch_size * (2**self.encoder.num_blocks)))
        self.decoder.input_shape = (H // self.patch_size, W // self.patch_size)

        x = self.patch_embed(x)

        x, skip_ftrs = self.encoder(x)

        x = self.bottleneck(x)

        x = self.decoder(x, skip_ftrs[::-1])

        x = self.final_expansion(x)

        x = self.head(x.permute(0, 3, 1, 2))  # Permute back to the appropriate shape

        return x


