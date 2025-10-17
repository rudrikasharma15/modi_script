"""
Multi-class Modi matra + character segmentation pipeline

Usage example (PowerShell):
python "matra_multiclass_segmentation.py" --data_dir "C:\path\to\modi_dataset" --out_dir "C:\path\to\results" --img_size 384 --epochs 40 --batch_size 8
"""
import os
import random
from pathlib import Path
import argparse
from glob import glob
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision import models
from torchvision.transforms import InterpolationMode
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam
import math

# -----------------------------
# PSEUDO-MASK GENERATION (matra vs character)
# -----------------------------
def remove_small_objects(bin_mask, min_size=20):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((bin_mask>0).astype(np.uint8), connectivity=8)
    out = np.zeros_like(bin_mask)
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area >= min_size:
            out[labels==lbl] = 255
    return out

def robust_pseudo_masks(img_gray, top_frac_guess=0.45):
    """
    Produce two-channel mask (matra, char) as uint8 0/255
    """
    if len(img_gray.shape) == 3:
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape
    # 1. Binarize (ink=255)
    _, bin_inv = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 2. Clean small noise
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    bin_clean = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, small_kernel, iterations=1)
    # 3. Accumulate multi-scale horizontal closings
    horiz_accum = np.zeros_like(bin_clean)
    for k_frac in [0.06, 0.12, 0.18]:
        horiz_len = max(6, int(w * k_frac))
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_len, 1))
        closed = cv2.morphologyEx(bin_clean, cv2.MORPH_CLOSE, horiz_kernel, iterations=1)
        horiz_accum = cv2.bitwise_or(horiz_accum, closed)
    # 4. Horizontal projection and headline candidate
    row_sums = horiz_accum.sum(axis=1).astype(np.float32)
    row_sums_sm = cv2.GaussianBlur(row_sums, (1,9), 0)
    top_limit = min(h, max(1, int(h * (top_frac_guess + 0.15))))
    top_index = int(np.argmax(row_sums_sm[:top_limit]))
    global_index = int(np.argmax(row_sums_sm))
    headline_row = top_index if row_sums_sm[top_index] > 0.6 * row_sums_sm[global_index] else global_index
    band_h = max(2, int(h * 0.06))
    band_start = max(0, headline_row - band_h)
    band_end = min(h, headline_row + band_h + 1)
    # 5. Component-based classification
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_clean, connectivity=8)
    matra_mask = np.zeros_like(bin_clean)
    char_mask = np.zeros_like(bin_clean)
    min_area = max(8, int(w * h * 0.0004))
    thin_ratio_thresh = 3.5
    for lbl in range(1, num_labels):
        x, y, ww, hh, area = stats[lbl]
        if area < min_area:
            continue
        intersects_headline = not (y + hh < band_start or y > band_end)
        wh_ratio = ww / max(1.0, hh)
        comp = (labels == lbl).astype(np.uint8) * 255
        if (wh_ratio >= thin_ratio_thresh and hh <= max(3, int(h * 0.03))) or intersects_headline:
            # assign component pixels to matra (not bounding box to preserve shape)
            matra_mask[labels==lbl] = 255
        else:
            char_mask[labels==lbl] = 255
    # 6. refine matra and char
    matra_mask = cv2.morphologyEx(matra_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9,1)), iterations=1)
    matra_mask = cv2.dilate(matra_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3,1)), iterations=1)
    residual = cv2.bitwise_and(bin_clean, cv2.bitwise_not(cv2.bitwise_or(matra_mask, char_mask)))
    char_mask = cv2.bitwise_or(char_mask, residual)
    char_mask = remove_small_objects(char_mask, min_area)
    matra_mask = (matra_mask>0).astype(np.uint8)*255
    char_mask = (char_mask>0).astype(np.uint8)*255
    # discard tiny matra
    if (matra_mask>0).mean() < 0.0005:
        matra_mask[:] = 0
    return np.stack([matra_mask, char_mask], axis=-1).astype(np.uint8)

# -----------------------------
# Dataset
# -----------------------------
class MultiClassMatraDataset(Dataset):
    def __init__(self, img_paths, resize=(384,384), transforms=None, pseudo_top_frac=0.45):
        self.paths = img_paths
        self.resize = resize
        self.transforms = transforms
        self.pseudo_top_frac = pseudo_top_frac

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        pil_gray = Image.open(p).convert("L")
        pil_resized_gray = pil_gray.resize(self.resize, Image.BILINEAR)
        img_np = np.array(pil_resized_gray)
        # generate pseudo two-channel mask
        mask_twoch = robust_pseudo_masks(img_np, top_frac_guess=self.pseudo_top_frac)  # (H,W,2) 0/255
        # convert to class map: 0 background, 1 character, 2 matra
        h,w,_ = mask_twoch.shape
        class_map = np.zeros((h,w), dtype=np.uint8)
        matra = mask_twoch[:,:,0] > 0
        char = mask_twoch[:,:,1] > 0
        # matra has priority (class 2), then char (1), else 0
        class_map[char] = 1
        class_map[matra] = 2
        # If all zero -> leave as background (helps model to learn absence)
        # Convert input to RGB for pretrained encoder
        pil_rgb = pil_resized_gray.convert("RGB")
        if self.transforms:
            img_tensor = self.transforms(pil_rgb)
        else:
            img_tensor = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            ])(pil_rgb)
        target = torch.from_numpy(class_map.astype(np.int64))
        return img_tensor, target, p

# -----------------------------
# Model (ResNet34 encoder U-Net)
# -----------------------------
class ResNet34UNetMultiClass(nn.Module):
    def __init__(self, pretrained=True, out_channels=3):
        super().__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
        self.conv1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # 64
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # 64
        self.layer2 = resnet.layer2  # 128
        self.layer3 = resnet.layer3  # 256
        self.layer4 = resnet.layer4  # 512

        def up_block(in_ch, out_ch):
            return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                nn.ReLU(inplace=True)
            )
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        self.up4 = up_block(512, 256)
        self.dec4 = conv_block(256+256, 256)
        self.up3 = up_block(256, 128)
        self.dec3 = conv_block(128+128, 128)
        self.up2 = up_block(128, 64)
        self.dec2 = conv_block(64+64, 64)
        self.up1 = up_block(64, 64)
        self.dec1 = conv_block(64+64, 64)
        self.final_up = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x0 = self.conv1(x)            # /2
        x1 = self.layer1(self.maxpool(x0))  # /4
        x2 = self.layer2(x1)         # /8
        x3 = self.layer3(x2)         # /16
        x4 = self.layer4(x3)         # /32

        u4 = self.up4(x4)
        d4 = self.dec4(torch.cat([u4, x3], dim=1))
        u3 = self.up3(d4)
        d3 = self.dec3(torch.cat([u3, x2], dim=1))
        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, x1], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, x0], dim=1))
        out = self.final_up(d1)
        out = self.final_conv(out)
        return out  # logits (B, C, H, W)

# -----------------------------
# Losses and metrics
# -----------------------------
class CrossEntropyDiceLoss(nn.Module):
    def __init__(self, weight=None, alpha=0.5, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        if weight is not None:
            self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        else:
            self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits, targets):
        # logits: (B,C,H,W), targets: (B,H,W) long
        ce_loss = self.ce(logits, targets)
        # Dice: compute on softmax-probs vs one-hot targets
        probs = torch.softmax(logits, dim=1)  # (B,C,H,W)
        # create one-hot target
        n_classes = logits.shape[1]
        target_onehot = torch.zeros_like(probs)
        # ignore_index handling: set mask
        valid_mask = (targets != 255)
        # avoid indexing issues
        targets_clamped = targets.clone()
        targets_clamped[~valid_mask] = 0
        target_onehot.scatter_(1, targets_clamped.unsqueeze(1), 1.0)
        target_onehot = target_onehot * valid_mask.unsqueeze(1).float()
        # per-class dice
        dims = (2,3)
        inter = (probs * target_onehot).sum(dim=dims)
        denom = probs.sum(dim=dims) + target_onehot.sum(dim=dims)
        dice_score = (2*inter + 1e-6) / (denom + 1e-6)  # (B,C)
        dice_loss = 1.0 - dice_score.mean()
        return self.alpha * ce_loss + (1.0 - self.alpha) * dice_loss

def dice_per_class(logits, targets, n_classes=3, eps=1e-6):
    probs = torch.softmax(logits, dim=1)
    target_onehot = torch.zeros_like(probs)
    valid_mask = (targets != 255)
    targets_clamped = targets.clone()
    targets_clamped[~valid_mask] = 0
    target_onehot.scatter_(1, targets_clamped.unsqueeze(1), 1.0)
    target_onehot = target_onehot * valid_mask.unsqueeze(1).float()
    inter = (probs * target_onehot).sum(dim=(0,2,3))
    denom = probs.sum(dim=(0,2,3)) + target_onehot.sum(dim=(0,2,3))
    dice = (2*inter + eps) / (denom + eps)
    return dice.cpu().numpy()  # shape (C,)

def iou_per_class(logits, targets, thresh=0.5, eps=1e-6):
    probs = torch.softmax(logits, dim=1)
    preds = (probs > thresh).float()
    # get per-class intersection and union over dataset batch
    n_classes = logits.shape[1]
    iou = []
    for c in range(n_classes):
        pred_c = preds[:,c,:,:]
        tgt_c = (targets == c).float()
        inter = (pred_c * tgt_c).sum(dim=(1,2))
        union = pred_c.sum(dim=(1,2)) + tgt_c.sum(dim=(1,2)) - inter
        iou_c = ((inter + eps) / (union + eps)).mean().item()
        iou.append(iou_c)
    return np.array(iou)

# -----------------------------
# Training utilities
# -----------------------------
scaler = GradScaler()

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for imgs, targets, _ in tqdm(loader, desc="Train", leave=False):
        imgs = imgs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        with autocast():
            logits = model(imgs)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.item()) * imgs.size(0)
    return total_loss / len(loader.dataset)

def validate(model, loader, device, thresh=0.5):
    model.eval()
    dices = []
    ious = []
    with torch.no_grad():
        for imgs, targets, _ in tqdm(loader, desc="Val", leave=False):
            imgs = imgs.to(device)
            targets = targets.to(device)
            logits = model(imgs)
            dice = dice_per_class(logits, targets, n_classes=logits.shape[1])
            iou = iou_per_class(logits, targets, thresh)
            dices.append(dice)
            ious.append(iou)
    dices = np.vstack(dices) if len(dices) else np.zeros((0, logits.shape[1]))
    ious = np.vstack(ious) if len(ious) else np.zeros((0, logits.shape[1]))
    mean_dice = np.mean(dices, axis=0) if dices.shape[0] else np.zeros((logits.shape[1],))
    mean_iou = np.mean(ious, axis=0) if ious.shape[0] else np.zeros((logits.shape[1],))
    return mean_dice, mean_iou

def overlay_and_save_multiclass(img_pil, pred_class_map, out_path):
    """
    pred_class_map: (H,W) int array with values 0,1,2
    Colors:
      0 background -> original
      1 character  -> green
      2 matra      -> red
    """
    rgb = np.array(img_pil.convert("RGB"))
    overlay = rgb.copy().astype(np.uint8)
    matra = (pred_class_map == 2)
    char = (pred_class_map == 1)
    overlay[matra] = [255, 0, 0]
    overlay[char] = [0, 255, 0]
    blended = cv2.addWeighted(overlay, 0.65, rgb, 0.35, 0)
    Image.fromarray(blended).save(out_path)

# -----------------------------
# Main
# -----------------------------
def main(args):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    data_root = Path(args.data_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_root / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    visuals_dir = out_root / "visuals"
    visuals_dir.mkdir(exist_ok=True)

    # collect images recursively
    exts = ("*.png","*.jpg","*.jpeg","*.bmp","*.tif")
    img_paths = []
    for e in exts:
        img_paths += sorted(map(str, data_root.rglob(e)))
    if not img_paths:
        print("No images found in", data_root)
        return
    random.shuffle(img_paths)
    n_val = max(1, int(len(img_paths) * args.val_frac))
    val_paths = img_paths[:n_val]
    train_paths = img_paths[n_val:]
    print(f"Found {len(img_paths)} images | Train: {len(train_paths)} | Val: {len(val_paths)}")

    resize = (args.img_size, args.img_size)
    train_tfms = T.Compose([
        T.Resize(resize, interpolation=InterpolationMode.BILINEAR),
        T.RandomHorizontalFlip(0.5),
        T.RandomRotation(5),
        T.ColorJitter(brightness=0.15, contrast=0.15),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    val_tfms = T.Compose([
        T.Resize(resize, interpolation=InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    train_ds = MultiClassMatraDataset(train_paths, resize=resize, transforms=train_tfms, pseudo_top_frac=args.top_frac)
    val_ds = MultiClassMatraDataset(val_paths, resize=resize, transforms=val_tfms, pseudo_top_frac=args.top_frac)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=max(1, args.num_workers//2), pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model = ResNet34UNetMultiClass(pretrained=True, out_channels=3).to(device)

    # class weights help with class imbalance. Default: background small, character medium, matra small
    if args.class_weights is None:
        weights = torch.tensor([1.0, 1.0, 2.0], dtype=torch.float32).to(device)  # emphasize matra a bit
    else:
        weights = torch.tensor(args.class_weights, dtype=torch.float32).to(device)
    criterion = CrossEntropyDiceLoss(weight=weights, alpha=args.ce_weight)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_score = -1.0
    for epoch in range(1, args.epochs+1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_dice, val_iou = validate(model, val_loader, device, thresh=args.thresh)
        avg_iou = float(np.mean(val_iou))
        scheduler.step()
        print(f"Epoch {epoch:03d} | TrainLoss: {tr_loss:.4f} | ValDice (BG/Char/Matra): {val_dice[0]:.4f}/{val_dice[1]:.4f}/{val_dice[2]:.4f} | ValIoU (BG/Char/Matra): {val_iou[0]:.4f}/{val_iou[1]:.4f}/{val_iou[2]:.4f}")
        if avg_iou > best_score:
            best_score = avg_iou
            torch.save(model.state_dict(), ckpt_dir / "best_model.pth")
            print(f"Saved best model (avg IoU={best_score:.4f})")
        if epoch % args.save_every == 0:
            torch.save(model.state_dict(), ckpt_dir / f"model_epoch_{epoch}.pth")

    # inference overlays on validation
    if (ckpt_dir / "best_model.pth").exists():
        model.load_state_dict(torch.load(ckpt_dir / "best_model.pth", map_location=device))
    model.eval()
    print("Generating overlays for validation images...")
    with torch.no_grad():
        for imgs, _, paths in tqdm(val_loader, desc="Visualize"):
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1).cpu().numpy()  # (B,3,H,W)
            for i, p in enumerate(paths):
                pil = Image.open(p).convert("L").resize(resize, Image.BILINEAR)
                pred_map = np.argmax(probs[i], axis=0).astype(np.uint8)
                out_p = visuals_dir / (Path(p).stem + "_overlay.png")
                overlay_and_save_multiclass(pil, pred_map, out_p)
    print("Done. Results and checkpoints saved to:", out_root)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./matra_multiclass_results")
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--val_frac", type=float, default=0.12)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--thresh", type=float, default=0.5)
    parser.add_argument("--ce_weight", type=float, default=0.5, help="alpha weight for CE vs Dice")
    parser.add_argument("--top_frac", type=float, default=0.45)
    parser.add_argument("--class_weights", nargs="+", type=float, default=None, help="optional class weights: bg char matra")
    args = parser.parse_args()
    main(args)


