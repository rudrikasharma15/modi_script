#!/usr/bin/env python3
r"""
Train matra-only segmentation using vowel folders only.
ResNet50 encoder + UNet-like decoder, AMP, IoU/Dice metrics,
and automatic matra mask generation focused on top/bottom regions.

Example usage (PowerShell):
python "C:\path\to\train_modi_matra_only.py" `
  --vowel_root "C:\Users\admin\Desktop\MODI_HChar\MODI_HChar\vowels" `
  --out_dir "C:\Users\admin\Desktop\MODI_HChar\matra_only_out" `
  --samples_per_class 1000 `
  --epochs 15 `
  --batch_size 16
"""

import os
import random
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision import models
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

# ------------------------------
# Utilities
# ------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(d):
    Path(d).mkdir(parents=True, exist_ok=True)

# ------------------------------
# Matra mask generator
# ------------------------------
def matra_mask_strict(img_gray, top_frac=0.25, bottom_frac=0.25, min_area=15, max_area_ratio=0.04):
    if len(img_gray.shape) == 3:
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape
    blur = cv2.GaussianBlur(img_gray, (3,3), 0)
    _, bin_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = np.zeros_like(bin_inv)
    top_band = bin_inv[:int(h*top_frac), :]
    bottom_band = bin_inv[int(h*(1-bottom_frac)):, :]

    def filter_band(band, y0):
        num, lbl, stats, _ = cv2.connectedComponentsWithStats(band)
        for i in range(1, num):
            x, y, bw, bh, area = stats[i]
            aspect = bw / (bh + 1e-6)
            if area < min_area or area > max_area_ratio*h*w or aspect > 8:
                continue
            mask[y0+y:y0+y+bh, x:x+bw] = 255
    filter_band(top_band, 0)
    filter_band(bottom_band, int(h*(1-bottom_frac)))

    # morphological cleanup (thinning)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (2,2)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (3,1)))
    return mask

# ------------------------------
# Overlay saving
# ------------------------------
def save_overlay(rgb_pil, mask_bin, out_path):
    rgb = np.array(rgb_pil.convert("RGB"))
    mask_resized = cv2.resize(mask_bin, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay = rgb.copy()
    overlay[mask_resized > 0] = [255, 0, 0]
    blended = cv2.addWeighted(overlay, 0.6, rgb, 0.4, 0)
    Image.fromarray(blended).save(out_path)

# ------------------------------
# Dataset building
# ------------------------------
def build_vowel_samples(vowel_root, samples_per_class=None, seed=42):
    vowel_root = Path(vowel_root)
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif"}
    all_samples = []
    rng = random.Random(seed)

    if not vowel_root.exists():
        raise FileNotFoundError(f"{vowel_root} not found")

    for sub in sorted([p for p in vowel_root.iterdir() if p.is_dir()]):
        imgs = [str(p) for p in sub.rglob("*") if p.suffix.lower() in exts]
        if len(imgs) == 0:
            continue
        rng.shuffle(imgs)
        if samples_per_class:
            imgs = imgs[:samples_per_class]
        all_samples += [(im, sub.name) for im in imgs]
    rng.shuffle(all_samples)
    print(f"Collected {len(all_samples)} vowel samples from {len(list(vowel_root.iterdir()))} vowel classes.")
    return all_samples

# ------------------------------
# Dataset
# ------------------------------
class MatraOnlyDataset(Dataset):
    def __init__(self, samples, resize=(320, 320), transforms=None, debug_save_mask_dir=None):
        self.samples = samples
        self.resize = resize
        self.transforms = transforms or T.Compose([
            T.Resize(resize),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        self.debug_save_mask_dir = debug_save_mask_dir
        if debug_save_mask_dir:
            ensure_dir(debug_save_mask_dir)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, cls = self.samples[idx]
        pil = Image.open(img_path).convert("L")
        pil_resized = pil.resize(self.resize, Image.BILINEAR)
        arr = np.array(pil_resized)
        mask = matra_mask_strict(arr)
        if self.debug_save_mask_dir and (idx % 200 == 0):
            dst = Path(self.debug_save_mask_dir) / f"{Path(img_path).stem}_mask.png"
            cv2.imwrite(str(dst), mask)

        img_rgb = pil_resized.convert("RGB")
        img_tensor = self.transforms(img_rgb)
        mask_tensor = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0)
        return img_tensor, mask_tensor, img_path

# ------------------------------
# Model
# ------------------------------
class ResNet50UNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        self.conv1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        def up_conv(in_c, out_c):
            return nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        self.up4, self.dec4 = up_conv(2048, 1024), conv_block(2048, 1024)
        self.up3, self.dec3 = up_conv(1024, 512), conv_block(1024, 512)
        self.up2, self.dec2 = up_conv(512, 256), conv_block(512, 256)
        self.up1, self.dec1 = up_conv(256, 64), conv_block(128, 64)
        self.final_up = nn.ConvTranspose2d(64, 64, 2, 2)
        self.final_conv = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x0 = self.conv1(x)
        x1 = self.layer1(self.maxpool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        u4 = self.up4(x4)
        d4 = self.dec4(torch.cat([u4, x3], dim=1))
        u3 = self.up3(d4)
        d3 = self.dec3(torch.cat([u3, x2], dim=1))
        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, x1], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, x0], dim=1))
        return self.final_conv(self.final_up(d1))

# ------------------------------
# Loss + metrics
# ------------------------------
class BCEDiceLoss(nn.Module):
    def __init__(self, bce_w=1.0, dice_w=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_w = bce_w
        self.dice_w = dice_w

    def forward(self, logits, targets):
        b = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum(dim=(2, 3))
        dice = (2 * inter + 1e-6) / (probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + 1e-6)
        return self.bce_w * b + self.dice_w * (1 - dice.mean())

def compute_metrics(logits, targets, thr=0.4):
    probs = torch.sigmoid(logits)
    preds = (probs > thr).float()
    tp = (preds * targets).sum(dim=(1, 2, 3))
    fp = (preds * (1 - targets)).sum(dim=(1, 2, 3))
    fn = ((1 - preds) * targets).sum(dim=(1, 2, 3))
    inter = tp
    union = (preds + targets - preds * targets).sum(dim=(1, 2, 3))
    iou = ((inter + 1e-6) / (union + 1e-6)).mean().item()
    dice = (2 * inter / (preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + 1e-6)).mean().item()
    return {"iou": iou, "dice": dice}

# ------------------------------
# Training & Validation
# ------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    running_loss = 0.0
    for imgs, masks, _ in tqdm(loader, desc="Train", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        with autocast():
            logits = model(imgs)
            loss = criterion(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)

def validate(model, loader, criterion, device, thr=0.4):
    model.eval()
    total_loss, metrics_list = 0.0, []
    with torch.no_grad():
        for imgs, masks, _ in tqdm(loader, desc="Val", leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            loss = criterion(logits, masks)
            total_loss += loss.item() * imgs.size(0)
            metrics_list.append(compute_metrics(logits, masks, thr))
    avg = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0]}
    return total_loss / len(loader.dataset), avg

# ------------------------------
# Main
# ------------------------------
def main(args):
    set_seed(args.seed)
    ensure_dir(args.out_dir)
    ensure_dir(os.path.join(args.out_dir, "checkpoints"))
    ensure_dir(os.path.join(args.out_dir, "visuals"))

    # ------------------ Dataset setup ------------------
    samples = build_vowel_samples(args.vowel_root, samples_per_class=args.samples_per_class, seed=args.seed)
    random.shuffle(samples)
    n_val = max(1, int(len(samples) * args.val_frac))
    val_samples, train_samples = samples[:n_val], samples[n_val:]
    print(f"Train: {len(train_samples)} | Val: {len(val_samples)}")

    resize = (args.img_size, args.img_size)
    train_tfms = T.Compose([
        T.RandomResizedCrop(resize, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(0.5),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_tfms = T.Compose([
        T.Resize(resize),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_ds = MatraOnlyDataset(train_samples, resize, train_tfms)
    val_ds = MatraOnlyDataset(val_samples, resize, val_tfms)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=max(1, args.num_workers // 2))

    # ------------------ Model setup ------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model = ResNet50UNet(pretrained=True).to(device)
    criterion = BCEDiceLoss(bce_w=1.0, dice_w=2.0)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler()

    best_iou = -1.0

    # ------------------ Training loop ------------------
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_dl, optimizer, criterion, device, scaler)
        val_loss, val_m = validate(model, val_dl, criterion, device, thr=args.save_thresh)
        scheduler.step()
        print(f"[{epoch}/{args.epochs}] Train: {tr_loss:.4f} | Val: {val_loss:.4f} | IoU: {val_m['iou']:.4f} | Dice: {val_m['dice']:.4f}")

        if val_m["iou"] > best_iou:
            best_iou = val_m["iou"]
            torch.save(model.state_dict(), os.path.join(args.out_dir, "checkpoints", "best_model.pth"))
            print(f"✅ New best IoU {best_iou:.4f} — model saved.")

    # ------------------ Overlay Generation ------------------
    print("Creating validation overlays (up to 200)...")
    model.eval()
    cnt = 0
    with torch.no_grad():
        for imgs, _, paths in tqdm(val_dl, desc="Overlay", leave=False):
            imgs = imgs.to(device)
            preds = torch.sigmoid(model(imgs)).cpu().numpy()

            for i, p in enumerate(paths):
                if cnt >= 200:
                    break

                # --- Threshold prediction ---
                mask = (preds[i, 0] > 0.5).astype(np.uint8)

                # --- Clean up mask: keep only top/bottom matras ---
                num, lbl, stats, _ = cv2.connectedComponentsWithStats(mask)
                clean = np.zeros_like(mask)
                h, w = mask.shape
                for j in range(1, num):
                    x, y, bw, bh, area = stats[j]
                    # keep only smaller top/bottom components
                    if area < 0.05 * (h * w) and (y < h * 0.4 or y > h * 0.6):
                        clean[lbl == j] = 1
                mask = clean

                # --- Morphological refine ---
                kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2))
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
                mask = (mask * 255).astype(np.uint8)

                # --- Save overlay ---
                pil = Image.open(p).convert("L").resize(resize)
                save_overlay(pil, mask, os.path.join(args.out_dir, "visuals", f"{Path(p).stem}_overlay_{cnt}.png"))
                cnt += 1

            if cnt >= 200:
                break
    print("✅ Done. Outputs in:", args.out_dir)


# ------------------------------ CLI ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vowel_root", required=True, type=str)
    parser.add_argument("--out_dir", default="./modi_matra_only_out", type=str)
    parser.add_argument("--samples_per_class", default=1000, type=int)
    parser.add_argument("--img_size", default=320, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--epochs", default=15, type=int)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--num_workers", default=6, type=int)
    parser.add_argument("--val_frac", default=0.05, type=float)
    parser.add_argument("--save_thresh", default=0.4, type=float)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    if args.samples_per_class is not None and args.samples_per_class <= 0:
        args.samples_per_class = None

    main(args)
