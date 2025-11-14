#!/usr/bin/env python3
"""
ResNet152-UNet full 4-class matra segmentation with improved auto-labeler.
0 = background, 1 = top matra, 2 = inline matra, 3 = bottom matra

Usage (single-line, PowerShell):
python "C:/Users/admin/Desktop/research2/train_modi_matra_resnet152_multiclass.py" --data_root "C:/Users/admin/Downloads/Dataset_Modi/Dataset_Modi" --out_dir "C:/Users/admin/Desktop/MODI_matra_out" --img_size 384 --epochs 20 --batch_size 6
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
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# ---------------- utils ----------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

# ---------------- improved auto-labeler ----------------
def generate_4class_mask_improved(gray):
    """
    Input: gray numpy array (H,W) or 3-channel image
    Output: uint8 mask (H,W) with values {0,1,2,3}
    """
    if len(gray.shape) == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # basic stroke extraction (strokes -> white)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    strokes = 255 - th

    # clean small noise but keep thin strokes
    strokes = cv2.morphologyEx(strokes, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))
    strokes = cv2.morphologyEx(strokes, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(3,1)))

    # detect potential headline using horizontal enhancement
    horiz_k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, w//8), 3))
    horiz = cv2.morphologyEx(strokes, cv2.MORPH_CLOSE, horiz_k)
    horiz = cv2.morphologyEx(horiz, cv2.MORPH_OPEN, horiz_k)

    n_h, lbl_h, stats_h, cent_h = cv2.connectedComponentsWithStats((horiz>0).astype(np.uint8), connectivity=8)
    headline_y = None
    best_bw = 0
    for i in range(1, n_h):
        x,y,bw,bh,area = stats_h[i]
        if bw > best_bw and bw > w*0.25 and bh < h*0.2:
            best_bw = bw
            headline_y = int(cent_h[i][1])

    # fallback body band by projection if no headline found
    proj = np.mean((strokes>0).astype(np.uint8), axis=1)
    proj_s = cv2.GaussianBlur((proj*255).astype(np.uint8),(11,1),0).astype(np.float32)/255.0
    proj_thr = max(0.05, proj_s.max()*0.45)
    rows = np.where(proj_s >= proj_thr)[0]
    if rows.size>0:
        b0,b1 = int(rows.min()), int(rows.max())
    else:
        b0,b1 = int(0.35*h), int(0.65*h)

    if headline_y is not None:
        band = max(2, int(h*0.08))
        body_y0 = max(0, headline_y - band)
        body_y1 = min(h-1, headline_y + band)
    else:
        body_y0, body_y1 = b0, b1

    # connected components on full strokes
    n, lbl, stats, cents = cv2.connectedComponentsWithStats((strokes>0).astype(np.uint8), connectivity=8)
    out = np.zeros((h,w), dtype=np.uint8)

    for i in range(1, n):
        x,y,bw,bh,area = stats[i]
        if area < 8:
            # very small: consider inline if overlapping body band modestly
            coords = np.where(lbl==i)[0]
            if coords.size>0:
                overlap = np.mean((coords >= body_y0) & (coords <= body_y1))
                if overlap > 0.25:
                    out[lbl==i] = 2
            continue

        cx, cy = cents[i]
        cy = float(cy)

        rows_i = np.unique(np.where(lbl==i)[0])
        overlap = np.mean((rows_i >= body_y0) & (rows_i <= body_y1)) if rows_i.size>0 else 0.0

        if overlap >= 0.4:
            cls = 2
        else:
            if cy < body_y0 - max(2, int(h*0.04)):
                cls = 1
            elif cy > body_y1 + max(2, int(h*0.04)):
                cls = 3
            else:
                cls = 2

        out[lbl==i] = cls

    # per-class morphological refine
    for c in (1,2,3):
        m = (out==c).astype(np.uint8)*255
        if m.sum()==0:
            continue
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(5,1)))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))
        out[m>0] = c

    return out

# ---------------- strict post-classifier for overlays ----------------
def classify_components_strict(mask_bin, orig_gray):
    """
    mask_bin: binary mask (0/255)
    orig_gray: np.array grayscale image
    returns class_mask (0/1/2/3)
    """
    if mask_bin.dtype != np.uint8:
        mb = (mask_bin>0).astype(np.uint8)*255
    else:
        mb = mask_bin.copy()
    h,w = mb.shape

    g = orig_gray.copy() if isinstance(orig_gray, np.ndarray) else np.array(orig_gray)
    blur = cv2.GaussianBlur(g, (3,3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    strokes = 255 - th
    horiz_k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, w//8), 3))
    horiz = cv2.morphologyEx(strokes, cv2.MORPH_CLOSE, horiz_k)
    n_h, lbl_h, stats_h, cent_h = cv2.connectedComponentsWithStats((horiz>0).astype(np.uint8), connectivity=8)
    headline_y = None
    for i in range(1, n_h):
        x,y,bw,bh,area = stats_h[i]
        if bw > w*0.25 and bh < h*0.15:
            headline_y = int(cent_h[i][1])
            break

    proj = np.mean((strokes>0).astype(np.uint8), axis=1)
    proj_s = cv2.GaussianBlur((proj*255).astype(np.uint8),(11,1),0).astype(np.float32)/255.0
    pr_thr = max(0.05, proj_s.max()*0.45)
    rows = np.where(proj_s >= pr_thr)[0]
    if rows.size>0:
        b0,b1 = int(rows.min()), int(rows.max())
    else:
        b0,b1 = int(0.35*h), int(0.65*h)

    if headline_y is not None:
        band = max(2, int(h*0.08))
        body_y0 = max(0, headline_y - band)
        body_y1 = min(h-1, headline_y + band)
    else:
        body_y0, body_y1 = b0, b1

    class_mask = np.zeros((h,w), dtype=np.uint8)
    n, lbl, stats, cents = cv2.connectedComponentsWithStats((mb>0).astype(np.uint8), connectivity=8)
    for i in range(1, n):
        x,y,bw,bh,area = stats[i]
        if area < 6:
            continue
        cx, cy = cents[i]; cy = float(cy)
        rows_i = np.unique(np.where(lbl==i)[0])
        overlap = np.mean((rows_i >= body_y0) & (rows_i <= body_y1)) if rows_i.size>0 else 0.0
        if overlap >= 0.35:
            cls = 2
        else:
            if cy < body_y0 - h*0.04:
                cls = 1
            elif cy > body_y1 + h*0.04:
                cls = 3
            else:
                cls = 2
        class_mask[lbl==i] = cls
    return class_mask

def save_overlay_colored(pil_gray, class_mask, out_path):
    rgb = np.array(pil_gray.convert("RGB"))
    h,w = rgb.shape[:2]
    mask_resized = cv2.resize(class_mask.astype(np.uint8), (w,h), interpolation=cv2.INTER_NEAREST)
    overlay = rgb.copy()
    overlay[mask_resized==1] = [255,0,0]
    overlay[mask_resized==2] = [0,255,0]
    overlay[mask_resized==3] = [0,0,255]
    blended = cv2.addWeighted(overlay, 0.6, rgb, 0.4, 0)
    Image.fromarray(blended).save(out_path)

# ---------------- Dataset ----------------
class Modi4ClsDataset(Dataset):
    def __init__(self, paths, img_size=384, transforms=None, debug_dir=None):
        self.paths = paths
        self.img_size = img_size
        self.transforms = transforms
        self.debug_dir = debug_dir
        if debug_dir:
            ensure_dir(debug_dir)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        pil = Image.open(p).convert("L")
        pil_r = pil.resize((self.img_size, self.img_size), Image.BILINEAR)
        arr = np.array(pil_r)
        mask = generate_4class_mask_improved(arr)    # uint8 {0..3}
        if self.debug_dir and (idx % 200 == 0):
            outp = Path(self.debug_dir) / f"{Path(p).stem}_labelgen.png"
            rgb = np.array(pil_r.convert("RGB"))
            overlay = rgb.copy()
            overlay[mask==1] = [255,0,0]; overlay[mask==2] = [0,255,0]; overlay[mask==3] = [0,0,255]
            Image.fromarray(overlay).save(outp)
        img = pil_r.convert("RGB")
        if self.transforms:
            img = self.transforms(img)
        else:
            tf = T.Compose([T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
            img = tf(img)
        mask_t = torch.from_numpy(mask.astype(np.int64))
        return img, mask_t, p

# ---------------- Model ----------------
class ResNet152_UNet_Multi(nn.Module):
    def __init__(self, pretrained=True, num_classes=4):
        super().__init__()
        resnet = models.resnet152(weights=models.ResNet152_Weights.DEFAULT if pretrained else None)
        self.conv1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        def upc(a,b): return nn.ConvTranspose2d(a,b,2,2)
        def block(a,b):
            return nn.Sequential(
                nn.Conv2d(a,b,3,padding=1,bias=False),
                nn.BatchNorm2d(b),
                nn.ReLU(inplace=True),
                nn.Conv2d(b,b,3,padding=1,bias=False),
                nn.BatchNorm2d(b),
                nn.ReLU(inplace=True),
            )
        self.up4 = upc(2048,1024); self.dec4 = block(2048,1024)
        self.up3 = upc(1024,512);  self.dec3 = block(1024,512)
        self.up2 = upc(512,256);   self.dec2 = block(512,256)
        self.up1 = upc(256,64);    self.dec1 = block(128,64)
        self.final_up = upc(64,64)
        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self,x):
        x0 = self.conv1(x)
        x1 = self.layer1(self.maxpool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        d4 = self.dec4(torch.cat([self.up4(x4), x3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), x2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), x1], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), x0], dim=1))
        out = self.final(self.final_up(d1))
        return out

# ---------------- Loss: weighted CE + Dice ----------------
class DiceLossMulti(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, logits, targets_onehot):
        probs = torch.softmax(logits, dim=1)
        dims = (0,2,3)
        inter = (probs * targets_onehot).sum(dims)
        union = (probs + targets_onehot).sum(dims)
        dice = (2*inter + self.eps) / (union + self.eps)
        return 1.0 - dice.mean()

class ComboLossMulti(nn.Module):
    def __init__(self, weight=None, dice_w=2.0, ce_w=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight) if weight is not None else nn.CrossEntropyLoss()
        self.dice = DiceLossMulti()
        self.ce_w = ce_w; self.dice_w = dice_w
    def forward(self, logits, targets):
        # logits (B,C,H,W), targets (B,H,W)
        ce_loss = self.ce(logits, targets)
        num_classes = logits.shape[1]
        tgt_onehot = torch.nn.functional.one_hot(targets, num_classes=num_classes).permute(0,3,1,2).float().to(logits.device)
        dice_loss = self.dice(logits, tgt_onehot)
        return self.ce_w * ce_loss + self.dice_w * dice_loss

# ---------------- metrics ----------------
def iou_per_class(pred, true, num_classes=4):
    pred = pred.view(-1).cpu().numpy()
    true = true.view(-1).cpu().numpy()
    ious = []
    for c in range(num_classes):
        p = (pred == c)
        t = (true == c)
        inter = (p & t).sum()
        union = p.sum() + t.sum() - inter
        if union == 0:
            ious.append(1.0)
        else:
            ious.append(inter/union)
    return np.array(ious)

# ---------------- train/val ----------------
def train_one_epoch(model, loader, optim, lossf, device, scaler):
    model.train()
    running = 0.0
    for imgs, masks, _ in tqdm(loader, desc="Train", leave=False):
        imgs = imgs.to(device); masks = masks.to(device)
        optim.zero_grad()
        with autocast():
            logits = model(imgs)
            loss = lossf(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        running += loss.item() * imgs.size(0)
    return running / len(loader.dataset)

def validate(model, loader, lossf, device):
    model.eval()
    running = 0.0
    iou_list = []
    with torch.no_grad():
        for imgs, masks, _ in tqdm(loader, desc="Val", leave=False):
            imgs = imgs.to(device); masks = masks.to(device)
            logits = model(imgs)
            loss = lossf(logits, masks)
            running += loss.item() * imgs.size(0)
            preds = torch.argmax(logits, dim=1)
            iou_list.append(iou_per_class(preds, masks))
    mean_iou = np.mean(iou_list, axis=0) if len(iou_list)>0 else np.zeros(4)
    return running / len(loader.dataset), mean_iou

# ---------------- collect images ----------------
def collect_image_paths(root, samples_per_class=None, exts={".png",".jpg",".jpeg",".bmp",".tif"}):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    all_imgs = []
    for sub in sorted([p for p in root.iterdir() if p.is_dir()]):
        imgs = [str(p) for p in sub.rglob("*") if p.suffix.lower() in exts]
        if len(imgs)==0:
            continue
        random.shuffle(imgs)
        if samples_per_class:
            imgs = imgs[:samples_per_class]
        all_imgs += imgs
    random.shuffle(all_imgs)
    return all_imgs

# ---------------- main ----------------
def main(args):
    set_seed(args.seed)
    ensure_dir(args.out_dir); ensure_dir(os.path.join(args.out_dir,"checkpoints")); ensure_dir(os.path.join(args.out_dir,"visuals"))
    ensure_dir(os.path.join(args.out_dir,"debug_labels"))

    samples = collect_image_paths(args.data_root, samples_per_class=args.samples_per_class)
    n_val = max(50, int(len(samples) * args.val_frac))
    val_samples, train_samples = samples[:n_val], samples[n_val:]
    print(f"Total {len(samples)}  Train {len(train_samples)}  Val {len(val_samples)}")

    train_tf = T.Compose([
        T.RandomResizedCrop((args.img_size,args.img_size), scale=(0.85,1.0)),
        T.RandomRotation(6),
        T.RandomHorizontalFlip(0.5),
        T.ColorJitter(0.12,0.12),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    val_tf = T.Compose([T.Resize((args.img_size,args.img_size)), T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    ds_train = Modi4ClsDataset(train_samples, args.img_size, train_tf, debug_dir=os.path.join(args.out_dir,"debug_labels"))
    ds_val = Modi4ClsDataset(val_samples, args.img_size, val_tf)

    train_dl = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_dl = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=max(1,args.num_workers//2), pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = ResNet152_UNet_Multi(pretrained=True).to(device)

    # suggested weights (tune if necessary) - more weight to top(1) and bottom(3)
    weights = torch.tensor([0.1, 3.0, 1.0, 2.5], dtype=torch.float).to(device)
    lossf = ComboLossMulti(weight=weights, dice_w=2.0, ce_w=1.0)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler()

    best_miou = -1.0
    for epoch in range(1, args.epochs+1):
        tr_loss = train_one_epoch(model, train_dl, optimizer, lossf, device, scaler)
        val_loss, val_iou = validate(model, val_dl, lossf, device)
        scheduler.step()
        miou = float(np.mean(val_iou))
        print(f"[{epoch}] TL {tr_loss:.4f}  VL {val_loss:.4f}  mIoU {miou:.4f}  clsIoU {val_iou.tolist()}")
        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), os.path.join(args.out_dir,"checkpoints","best_model.pth"))
            print("Saved best.")

    # generate overlays on validation set using strict classifier
    print("Generating overlays on validation set...")
    model.eval()
    cnt = 0
    with torch.no_grad():
        for img_t, _, paths in tqdm(val_dl, desc="Overlay", leave=False):
            img_t = img_t.to(device)
            logits = model(img_t)
            preds = torch.argmax(logits, dim=1).cpu().numpy()  # (B,H,W)
            for i, p in enumerate(paths):
                if cnt >= args.max_overlays:
                    break
                mask4 = preds[i].astype(np.uint8)
                # refine with strict component classifier applied to predicted inline/top/bottom combined binary
                binary = (mask4 > 0).astype(np.uint8)*255
                orig = Image.open(p).convert("L").resize((args.img_size, args.img_size))
                class_mask = classify_components_strict(binary, np.array(orig))
                save_overlay_colored(orig, class_mask, os.path.join(args.out_dir,"visuals", f"{Path(p).stem}_overlay_{cnt}.png"))
                cnt += 1
            if cnt >= args.max_overlays:
                break
    print("Done. Outputs:", args.out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, type=str)
    parser.add_argument("--out_dir", default="./modi_matra_out", type=str)
    parser.add_argument("--samples_per_class", default=None, type=int)
    parser.add_argument("--img_size", default=384, type=int)
    parser.add_argument("--batch_size", default=6, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--num_workers", default=6, type=int)
    parser.add_argument("--val_frac", default=0.06, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--thr", default=0.45, type=float)
    parser.add_argument("--max_overlays", default=300, type=int)
    args = parser.parse_args()
    if args.samples_per_class is not None and args.samples_per_class <= 0:
        args.samples_per_class = None
    main(args)
