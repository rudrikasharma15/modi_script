#!/usr/bin/env python3
"""
Enhanced ResNet152-UNet for Modi Script Matra Segmentation
Improved for non-straight scripts with advanced preprocessing and post-processing
0 = background, 1 = top matra, 2 = inline matra, 3 = bottom matra
"""
import os
import random
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
from scipy import ndimage
from skimage import morphology, measure

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision import models
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# ================ UTILITIES ================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

# ================ ADVANCED PREPROCESSING ================
def adaptive_binarization(gray):
    """Better binarization for non-straight scripts"""
    # Adaptive thresholding
    binary1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 21, 10)
    
    # Otsu's thresholding
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, binary2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Combine both methods
    binary = cv2.bitwise_or(binary1, binary2)
    
    # Clean noise
    kernel = np.ones((2,2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    return binary

def detect_baseline_advanced(binary, orig_gray):
    """Advanced baseline detection for curved/non-straight Modi scripts"""
    h, w = binary.shape
    
    # Method 1: Horizontal projection with smoothing
    proj = np.sum(binary, axis=1)
    proj_smooth = ndimage.gaussian_filter1d(proj, sigma=h//30)
    
    # Find peaks in projection (body region)
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(proj_smooth, height=w*0.15, distance=h//10)
    
    if len(peaks) > 0:
        # Main body is around highest peak
        main_peak = peaks[np.argmax(proj_smooth[peaks])]
        body_center = main_peak
    else:
        body_center = h // 2
    
    # Method 2: Find horizontal strokes (potential baseline indicators)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w//6, 3))
    horiz_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel)
    
    # Get connected components of horizontal lines
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(horiz_lines, connectivity=8)
    
    baseline_candidates = []
    for i in range(1, num_labels):
        x, y, width, height, area = stats[i]
        if width > w * 0.25 and height < h * 0.15:
            cy = int(centroids[i][1])
            baseline_candidates.append(cy)
    
    # Method 3: Skeleton-based centerline detection
    skeleton = morphology.skeletonize(binary > 0)
    skel_proj = np.sum(skeleton, axis=1)
    skel_smooth = ndimage.gaussian_filter1d(skel_proj, sigma=h//25)
    
    # Combine all methods
    if baseline_candidates:
        baseline_y = int(np.median(baseline_candidates))
    else:
        baseline_y = body_center
    
    # Define body band with adaptive width
    band_width = max(int(h * 0.12), 15)
    body_top = max(0, baseline_y - band_width)
    body_bottom = min(h - 1, baseline_y + band_width)
    
    return body_top, body_bottom, baseline_y

def classify_component_by_position(component_mask, body_top, body_bottom, baseline_y, h, w):
    """
    Classify a component based on its position and shape
    Returns: 1 (top), 2 (inline), or 3 (bottom)
    """
    # Get component properties
    ys, xs = np.where(component_mask)
    
    if len(ys) == 0:
        return 0
    
    cy = np.mean(ys)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    
    comp_height = y_max - y_min + 1
    comp_width = x_max - x_min + 1
    
    # Calculate overlap with body region
    body_overlap_pixels = np.sum((ys >= body_top) & (ys <= body_bottom))
    total_pixels = len(ys)
    overlap_ratio = body_overlap_pixels / total_pixels if total_pixels > 0 else 0
    
    # Aspect ratio and shape features
    aspect_ratio = comp_width / (comp_height + 1e-6)
    
    # Multi-factor classification
    
    # Strong overlap with body = inline
    if overlap_ratio > 0.5:
        return 2
    
    # Component mostly above body
    if y_max < body_top - h*0.03:
        return 1
    
    # Component mostly below body
    if y_min > body_bottom + h*0.03:
        return 3
    
    # Center-based classification with margin
    margin = h * 0.08
    
    if cy < body_top - margin:
        return 1
    elif cy > body_bottom + margin:
        return 3
    else:
        # Ambiguous region - use additional features
        
        # Horizontal strokes above baseline likely top matras
        if aspect_ratio > 2.5 and cy < baseline_y - margin/2:
            return 1
        
        # Curved components below baseline likely bottom matras
        if aspect_ratio < 1.5 and cy > baseline_y + margin/2:
            return 3
        
        return 2

# ================ ENHANCED AUTO-LABELER ================
def generate_4class_mask_enhanced(gray):
    """Enhanced auto-labeler with better handling of Modi script characteristics"""
    if len(gray.shape) == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    
    h, w = gray.shape
    
    # 1) Advanced binarization
    binary = adaptive_binarization(gray)
    
    # 2) Advanced baseline detection
    body_top, body_bottom, baseline_y = detect_baseline_advanced(binary, gray)
    
    # 3) Connected component analysis with size filtering
    # Remove very small noise
    cleaned = morphology.remove_small_objects(binary > 0, min_size=max(6, h*w//5000))
    cleaned = (cleaned * 255).astype(np.uint8)
    
    # Get connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    
    # 4) Classify each component
    output_mask = np.zeros((h, w), dtype=np.uint8)
    
    for i in range(1, num_labels):
        x, y, width, height, area = stats[i]
        
        # Skip very small components
        if area < max(6, h*w//8000):
            continue
        
        # Get component mask
        comp_mask = (labels == i)
        
        # Classify component
        cls = classify_component_by_position(comp_mask, body_top, body_bottom, 
                                             baseline_y, h, w)
        
        output_mask[comp_mask] = cls
    
    # 5) Post-processing: merge nearby components of same class
    for cls in [1, 2, 3]:
        cls_mask = (output_mask == cls).astype(np.uint8)
        if cls_mask.sum() == 0:
            continue
        
        # Slight dilation to connect nearby parts
        kernel_size = max(2, min(5, int(h * 0.02)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated = cv2.dilate(cls_mask, kernel, iterations=1)
        
        # Re-label and filter
        dilated = morphology.remove_small_objects(dilated > 0, min_size=max(4, h*w//10000))
        output_mask[dilated & (output_mask == 0)] = cls
    
    return output_mask

# ================ STRICT POST-CLASSIFIER ================
def classify_components_strict(mask_bin, orig_gray):
    """Strict post-classification for predicted masks"""
    if mask_bin.dtype != np.uint8:
        mb = (mask_bin > 0).astype(np.uint8) * 255
    else:
        mb = mask_bin.copy()
    
    h, w = mb.shape
    
    # Use enhanced detection
    binary = adaptive_binarization(orig_gray)
    body_top, body_bottom, baseline_y = detect_baseline_advanced(binary, orig_gray)
    
    # Classify components in predicted mask
    class_mask = np.zeros((h, w), dtype=np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mb, connectivity=8)
    
    for i in range(1, num_labels):
        x, y, width, height, area = stats[i]
        
        if area < 6:
            continue
        
        comp_mask = (labels == i)
        cls = classify_component_by_position(comp_mask, body_top, body_bottom, 
                                             baseline_y, h, w)
        
        class_mask[comp_mask] = cls
    
    return class_mask

def save_overlay_colored(pil_gray, class_mask, out_path):
    """Save colored overlay visualization"""
    rgb = np.array(pil_gray.convert("RGB"))
    h, w = rgb.shape[:2]
    mask_resized = cv2.resize(class_mask.astype(np.uint8), (w, h), 
                              interpolation=cv2.INTER_NEAREST)
    
    overlay = rgb.copy()
    # Red for top matras
    overlay[mask_resized == 1] = [255, 0, 0]
    # Green for inline matras
    overlay[mask_resized == 2] = [0, 255, 0]
    # Blue for bottom matras
    overlay[mask_resized == 3] = [0, 0, 255]
    
    blended = cv2.addWeighted(overlay, 0.6, rgb, 0.4, 0)
    Image.fromarray(blended).save(out_path)

# ================ DATASET ================
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
        
        # Use enhanced auto-labeler
        mask = generate_4class_mask_enhanced(arr)
        
        # Debug visualization
        if self.debug_dir and (idx % 100 == 0):
            outp = Path(self.debug_dir) / f"{Path(p).stem}_labelgen.png"
            save_overlay_colored(pil_r, mask, outp)
        
        img = pil_r.convert("RGB")
        if self.transforms:
            img = self.transforms(img)
        else:
            tf = T.Compose([
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            img = tf(img)
        
        mask_t = torch.from_numpy(mask.astype(np.int64))
        return img, mask_t, p

# ================ MODEL ================
class ResNet152_UNet_Multi(nn.Module):
    def __init__(self, pretrained=True, num_classes=4):
        super().__init__()
        resnet = models.resnet152(
            weights=models.ResNet152_Weights.DEFAULT if pretrained else None
        )
        
        self.conv1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        def upc(a, b):
            return nn.ConvTranspose2d(a, b, 2, 2)
        
        def block(a, b):
            return nn.Sequential(
                nn.Conv2d(a, b, 3, padding=1, bias=False),
                nn.BatchNorm2d(b),
                nn.ReLU(inplace=True),
                nn.Conv2d(b, b, 3, padding=1, bias=False),
                nn.BatchNorm2d(b),
                nn.ReLU(inplace=True),
            )
        
        self.up4 = upc(2048, 1024)
        self.dec4 = block(2048, 1024)
        self.up3 = upc(1024, 512)
        self.dec3 = block(1024, 512)
        self.up2 = upc(512, 256)
        self.dec2 = block(512, 256)
        self.up1 = upc(256, 64)
        self.dec1 = block(128, 64)
        self.final_up = upc(64, 64)
        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
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

# ================ LOSS FUNCTIONS ================
class DiceLossMulti(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    
    def forward(self, logits, targets_onehot):
        probs = torch.softmax(logits, dim=1)
        dims = (0, 2, 3)
        inter = (probs * targets_onehot).sum(dims)
        union = (probs + targets_onehot).sum(dims)
        dice = (2 * inter + self.eps) / (union + self.eps)
        return 1.0 - dice.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none', 
                                              weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

class ComboLossMulti(nn.Module):
    def __init__(self, weight=None, dice_w=3.0, focal_w=1.0):
        super().__init__()
        self.focal = FocalLoss(alpha=weight, gamma=2.0)
        self.dice = DiceLossMulti()
        self.focal_w = focal_w
        self.dice_w = dice_w
    
    def forward(self, logits, targets):
        focal_loss = self.focal(logits, targets)
        
        num_classes = logits.shape[1]
        tgt_onehot = torch.nn.functional.one_hot(targets, num_classes=num_classes)
        tgt_onehot = tgt_onehot.permute(0, 3, 1, 2).float().to(logits.device)
        dice_loss = self.dice(logits, tgt_onehot)
        
        return self.focal_w * focal_loss + self.dice_w * dice_loss

# ================ METRICS ================
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
            ious.append(inter / union)
    
    return np.array(ious)

# ================ TRAINING ================
def train_one_epoch(model, loader, optim, lossf, device, scaler):
    model.train()
    running = 0.0
    
    for imgs, masks, _ in tqdm(loader, desc="Train", leave=False):
        imgs = imgs.to(device)
        masks = masks.to(device)
        
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
            imgs = imgs.to(device)
            masks = masks.to(device)
            
            logits = model(imgs)
            loss = lossf(logits, masks)
            running += loss.item() * imgs.size(0)
            
            preds = torch.argmax(logits, dim=1)
            iou_list.append(iou_per_class(preds, masks))
    
    mean_iou = np.mean(iou_list, axis=0) if len(iou_list) > 0 else np.zeros(4)
    return running / len(loader.dataset), mean_iou

# ================ DATA COLLECTION ================
def collect_image_paths(root, samples_per_class=None, 
                        exts={".png", ".jpg", ".jpeg", ".bmp", ".tif"}):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    
    all_imgs = []
    for sub in sorted([p for p in root.iterdir() if p.is_dir()]):
        imgs = [str(p) for p in sub.rglob("*") if p.suffix.lower() in exts]
        if len(imgs) == 0:
            continue
        
        random.shuffle(imgs)
        if samples_per_class:
            imgs = imgs[:samples_per_class]
        all_imgs += imgs
    
    random.shuffle(all_imgs)
    return all_imgs

# ================ MAIN ================
def main(args):
    set_seed(args.seed)
    
    # Create output directories
    ensure_dir(args.out_dir)
    ensure_dir(os.path.join(args.out_dir, "checkpoints"))
    ensure_dir(os.path.join(args.out_dir, "visuals"))
    ensure_dir(os.path.join(args.out_dir, "debug_labels"))

    # Collect and split data
    samples = collect_image_paths(args.data_root, 
                                  samples_per_class=args.samples_per_class)
    n_val = max(50, int(len(samples) * args.val_frac))
    val_samples, train_samples = samples[:n_val], samples[n_val:]
    
    print(f"Total: {len(samples)}  Train: {len(train_samples)}  Val: {len(val_samples)}")

    # Data augmentation
    train_tf = T.Compose([
        T.RandomResizedCrop((args.img_size, args.img_size), scale=(0.88, 1.0)),
        T.RandomRotation(8),
        T.RandomHorizontalFlip(0.5),
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_tf = T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create datasets
    ds_train = Modi4ClsDataset(train_samples, args.img_size, train_tf,
                               debug_dir=os.path.join(args.out_dir, "debug_labels"))
    ds_val = Modi4ClsDataset(val_samples, args.img_size, val_tf)

    # Create data loaders
    train_dl = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True)
    val_dl = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                        num_workers=max(1, args.num_workers // 2), pin_memory=True)

    # Setup device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = ResNet152_UNet_Multi(pretrained=True).to(device)

    # Class weights (emphasize top and bottom matras)
    weights = torch.tensor([0.1, 4.0, 1.0, 3.5], dtype=torch.float).to(device)
    lossf = ComboLossMulti(weight=weights, dice_w=3.0, focal_w=1.0)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler()

    # Training loop
    best_miou = -1.0
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_dl, optimizer, lossf, device, scaler)
        val_loss, val_iou = validate(model, val_dl, lossf, device)
        scheduler.step()
        
        miou = float(np.mean(val_iou))
        print(f"[Epoch {epoch}] Train Loss: {tr_loss:.4f}  Val Loss: {val_loss:.4f}")
        print(f"  mIoU: {miou:.4f}  Class IoUs: {val_iou.tolist()}")
        
        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(),
                      os.path.join(args.out_dir, "checkpoints", "best_model.pth"))
            print("  ✓ Saved best model")

    # Generate overlays on validation set
    print("\nGenerating overlays on validation set...")
    model.eval()
    cnt = 0
    
    with torch.no_grad():
        for img_t, _, paths in tqdm(val_dl, desc="Overlay", leave=False):
            img_t = img_t.to(device)
            logits = model(img_t)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            for i, p in enumerate(paths):
                if cnt >= args.max_overlays:
                    break
                
                mask4 = preds[i].astype(np.uint8)
                binary = (mask4 > 0).astype(np.uint8) * 255
                
                orig = Image.open(p).convert("L").resize((args.img_size, args.img_size))
                class_mask = classify_components_strict(binary, np.array(orig))
                
                save_overlay_colored(
                    orig, class_mask,
                    os.path.join(args.out_dir, "visuals", 
                                f"{Path(p).stem}_overlay_{cnt}.png")
                )
                cnt += 1
            
            if cnt >= args.max_overlays:
                break
    
    print(f"\nDone! Outputs saved to: {args.out_dir}")
    print(f"Best mIoU achieved: {best_miou:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Modi Matra Segmentation")
    parser.add_argument("--data_root", required=True, type=str,
                       help="Root directory containing Modi script images")
    parser.add_argument("--out_dir", default="./modi_matra_out_enhanced", type=str,
                       help="Output directory")
    parser.add_argument("--samples_per_class", default=None, type=int,
                       help="Limit samples per class subdirectory")
    parser.add_argument("--img_size", default=384, type=int,
                       help="Input image size")
    parser.add_argument("--batch_size", default=6, type=int,
                       help="Batch size")
    parser.add_argument("--epochs", default=25, type=int,
                       help="Number of training epochs")
    parser.add_argument("--lr", default=2e-4, type=float,
                       help="Learning rate")
    parser.add_argument("--num_workers", default=6, type=int,
                       help="Number of data loader workers")
    parser.add_argument("--val_frac", default=0.06, type=float,
                       help="Validation set fraction")
    parser.add_argument("--seed", default=42, type=int,
                       help="Random seed")
    parser.add_argument("--max_overlays", default=300, type=int,
                       help="Maximum overlay visualizations to generate")
    
    args = parser.parse_args()
    
    if args.samples_per_class is not None and args.samples_per_class <= 0:
        args.samples_per_class = None
    
    main(args)
