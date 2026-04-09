"""
Encoder registry for stereo matching experiments.

Usage:
    from models import load_encoder, extract_features, ENCODER_LIST

    model, meta = load_encoder("DINOv3 ViT-B", device="cpu")
    feat = extract_features(model, meta, img_tensor, hp, wp)
    # feat: (D, hp, wp) tensor on CPU
"""
import sys
import types
import warnings
from pathlib import Path

import torch
import timm

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).parent
CROCO_DIR  = SCRIPT_DIR / "croco"
CROCO_CKPT = SCRIPT_DIR / "CroCo_V2_ViTBase_SmallDecoder.pth"

ENCODER_LIST = [
    "DINOv3 ViT-B",
    "MAE ViT-B",
    "DINOv2 ViT-B",
    "DINOv2+Reg ViT-B",
    "DINOv1 ViT-B",
    "CLIP ViT-B",
    "Supervised ViT-B",
    "CroCo v2 ViT-B",
    "Random ViT-B",
]

# Static metadata (training regime, positional encoding, patch size)
ENCODER_META_STATIC = {
    "DINOv3 ViT-B":     {"training": "DINO v3",     "pos_embed": "RoPE",    "patch_size": 16},
    "MAE ViT-B":        {"training": "MAE",          "pos_embed": "Learned", "patch_size": 16},
    "DINOv2 ViT-B":     {"training": "DINO v2",      "pos_embed": "Learned", "patch_size": 14},
    "DINOv2+Reg ViT-B": {"training": "DINO v2+Reg",  "pos_embed": "Learned", "patch_size": 14},
    "DINOv1 ViT-B":     {"training": "DINO v1",      "pos_embed": "Learned", "patch_size": 16},
    "CLIP ViT-B":       {"training": "CLIP",         "pos_embed": "Learned", "patch_size": 16},
    "Supervised ViT-B": {"training": "Supervised",   "pos_embed": "Learned", "patch_size": 16},
    "CroCo v2 ViT-B":   {"training": "CroCo v2",    "pos_embed": "RoPE",    "patch_size": 16},
    "Random ViT-B":     {"training": "Random",       "pos_embed": "RoPE",    "patch_size": 16},
}

_TIMM_SPECS = {
    "DINOv3 ViT-B": dict(
        timm_id="vit_base_patch16_dinov3.lvd1689m",
        kwargs=dict(pretrained=True, num_classes=0),
    ),
    "MAE ViT-B": dict(
        timm_id="vit_base_patch16_224.mae",
        kwargs=dict(pretrained=True, num_classes=0, dynamic_img_size=True),
    ),
    "DINOv2 ViT-B": dict(
        timm_id="vit_base_patch14_dinov2.lvd142m",
        kwargs=dict(pretrained=True, num_classes=0, dynamic_img_size=True),
    ),
    "DINOv2+Reg ViT-B": dict(
        timm_id="vit_base_patch14_reg4_dinov2.lvd142m",
        kwargs=dict(pretrained=True, num_classes=0, dynamic_img_size=True),
    ),
    "DINOv1 ViT-B": dict(
        timm_id="vit_base_patch16_224.dino",
        kwargs=dict(pretrained=True, num_classes=0, dynamic_img_size=True),
    ),
    "CLIP ViT-B": dict(
        timm_id="vit_base_patch16_clip_224.openai",
        kwargs=dict(pretrained=True, num_classes=0, dynamic_img_size=True),
    ),
    "Supervised ViT-B": dict(
        timm_id="vit_base_patch16_224.augreg_in21k_ft_in1k",
        kwargs=dict(pretrained=True, num_classes=0, dynamic_img_size=True),
    ),
    "Random ViT-B": dict(
        timm_id="vit_base_patch16_dinov3.lvd1689m",
        kwargs=dict(pretrained=False, num_classes=0),
    ),
}


def _load_croco(device: str):
    import subprocess
    import urllib.request

    if not CROCO_DIR.exists():
        print("  Cloning naver/croco …")
        subprocess.run(
            ["git", "clone", "https://github.com/naver/croco.git", str(CROCO_DIR)],
            check=True, capture_output=True,
        )
    if str(CROCO_DIR) not in sys.path:
        sys.path.insert(0, str(CROCO_DIR))
    if not CROCO_CKPT.exists():
        print("  Downloading CroCo v2 checkpoint (~200 MB) …")
        url = ("https://download.europe.naverlabs.com/ComputerVision/"
               "CroCo/CroCo_V2_ViTBase_SmallDecoder.pth")
        urllib.request.urlretrieve(url, CROCO_CKPT)

    from models.croco import CroCoNet  # noqa: PLC0415

    ckpt  = torch.load(CROCO_CKPT, map_location="cpu")
    model = CroCoNet(**ckpt.get("croco_kwargs", {}))
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval().to(device)

    def _flex_forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

    model.patch_embed.forward = types.MethodType(_flex_forward, model.patch_embed)
    return model, model.enc_embed_dim


def load_encoder(name: str, device: str = "cpu"):
    """
    Load a named encoder.
    Returns (model, meta_dict) where meta contains:
        name, training, pos_embed, patch_size, feat_dim, _type
    """
    if name == "CroCo v2 ViT-B":
        model, feat_dim = _load_croco(device)
        meta = {**ENCODER_META_STATIC[name], "feat_dim": feat_dim,
                "name": name, "_type": "croco"}
    elif name in _TIMM_SPECS:
        spec  = _TIMM_SPECS[name]
        model = timm.create_model(spec["timm_id"], **spec["kwargs"]).eval().to(device)
        meta  = {**ENCODER_META_STATIC[name], "feat_dim": model.embed_dim,
                 "name": name, "_type": "timm"}
    else:
        raise ValueError(f"Unknown encoder: {name!r}. Available: {ENCODER_LIST}")
    return model, meta


def extract_features(model, meta: dict, img_tensor: torch.Tensor,
                     hp: int, wp: int) -> torch.Tensor:
    """
    Extract patch features from a (3, H, W) image tensor.
    Returns (D, hp, wp) on CPU.
    """
    n = hp * wp
    device = next(model.parameters()).device
    x = img_tensor.unsqueeze(0).to(device)

    with torch.inference_mode():
        if meta["_type"] == "croco":
            feat = model.patch_embed(x)
            gy, gx = torch.meshgrid(
                torch.arange(hp, device=device),
                torch.arange(wp, device=device),
                indexing="ij",
            )
            xpos = torch.stack([gy, gx], dim=-1).long().reshape(1, -1, 2)
            for blk in model.enc_blocks:
                feat = blk(feat, xpos=xpos)
            feat   = model.enc_norm(feat)
            tokens = feat.squeeze(0)            # (N, D)
        else:
            out    = model.forward_features(x)  # (1, prefix+N, D)
            tokens = out[:, -n:, :].squeeze(0)  # (N, D)

    D = meta["feat_dim"]
    return tokens.reshape(hp, wp, D).permute(2, 0, 1).cpu()  # (D, hp, wp)
