#!/usr/bin/env python3
# ===============================================================
# 🌍 Remote Sensing Segmentation Training
# Models: UNet, SegNet, DeepLabV3
# ===============================================================

import os, torch, numpy as np, matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from pathlib import Path

# ===============================================================
# Dataset
# ===============================================================

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

def list_files(root: Path):
    return [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]

class LandCoverDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=(256,256), num_classes=6):
        self.img_dir, self.mask_dir = Path(img_dir), Path(mask_dir)
        self.images = sorted(list_files(self.img_dir))
        self.masks = sorted(list_files(self.mask_dir))
        if len(self.images) == 0:
            raise RuntimeError(f"No images found in {img_dir}")
        self.img_size = img_size
        self.num_classes = num_classes
        self.resize_img = transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR)
        self.resize_mask = transforms.Resize(img_size, interpolation=transforms.InterpolationMode.NEAREST)
        self.tf = transforms.ToTensor()

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        mask = Image.open(self.masks[idx]).convert("L")
        img = self.resize_img(img)
        mask = self.resize_mask(mask)
        img = self.tf(img)
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))

        mask[mask == 255] = 0
        mask = mask - mask.min()
        mask = torch.clamp(mask, 0, self.num_classes - 1)
        return img, mask

# ===============================================================
# Model Definitions
# ===============================================================

class UNetBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU()
        )
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=6):
        super().__init__()
        self.enc1 = UNetBlock(in_ch, 64)
        self.enc2 = UNetBlock(64, 128)
        self.enc3 = UNetBlock(128, 256)
        self.enc4 = UNetBlock(256, 512)
        self.center = UNetBlock(512, 1024)
        self.pool = nn.MaxPool2d(2)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = UNetBlock(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = UNetBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = UNetBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = UNetBlock(128, 64)
        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        c = self.center(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(c), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)

class SegNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=6):
        super().__init__()
        self.enc1 = self._block(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2,2,return_indices=True)
        self.enc2 = self._block(64,128)
        self.pool2 = nn.MaxPool2d(2,2,return_indices=True)
        self.enc3 = self._block(128,256)
        self.pool3 = nn.MaxPool2d(2,2,return_indices=True)
        self.unpool3 = nn.MaxUnpool2d(2,2)
        self.dec3 = self._block(256,128)
        self.unpool2 = nn.MaxUnpool2d(2,2)
        self.dec2 = self._block(128,64)
        self.unpool1 = nn.MaxUnpool2d(2,2)
        self.final = nn.Conv2d(64, num_classes, 1)

    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c,out_c,3,padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c,out_c,3,padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.enc1(x); x1_size = x.size(); x, idx1 = self.pool1(x)
        x = self.enc2(x); x2_size = x.size(); x, idx2 = self.pool2(x)
        x = self.enc3(x); x3_size = x.size(); x, idx3 = self.pool3(x)
        x = self.unpool3(x, idx3, output_size=x3_size); x = self.dec3(x)
        x = self.unpool2(x, idx2, output_size=x2_size); x = self.dec2(x)
        x = self.unpool1(x, idx1, output_size=x1_size); x = self.final(x)
        return x

def get_deeplab(num_classes=6):
    model = models.segmentation.deeplabv3_resnet50(pretrained=False)
    model.classifier[4] = nn.Conv2d(256, num_classes, 1)
    return model

# ===============================================================
# Training Helpers
# ===============================================================

def fast_hist(true, pred, n_class):
    k = (true >= 0) & (true < n_class)
    return np.bincount(n_class * true[k].astype(int) + pred[k].astype(int),
                       minlength=n_class**2).reshape(n_class, n_class)

def compute_miou(conf_mat):
    inter = np.diag(conf_mat)
    union = conf_mat.sum(1)+conf_mat.sum(0)-inter
    return np.nanmean(inter/np.maximum(union,1))

def train_one_epoch(model, loader, opt, crit, device, num_classes):
    model.train(); total_loss=0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        opt.zero_grad()
        out = model(imgs)
        if isinstance(out, dict): out = out['out']
        loss = crit(out, masks)
        loss.backward(); opt.step()
        total_loss += loss.item()*imgs.size(0)
    return total_loss/len(loader.dataset)

@torch.no_grad()
def validate(model, loader, crit, device, num_classes):
    model.eval(); conf=np.zeros((num_classes,num_classes))
    val_loss=0
    for imgs,masks in loader:
        imgs,masks=imgs.to(device),masks.to(device)
        out=model(imgs)
        if isinstance(out, dict): out=out['out']
        loss=crit(out,masks); val_loss+=loss.item()*imgs.size(0)
        preds=torch.argmax(out,1)
        conf+=fast_hist(masks.cpu().numpy().flatten(), preds.cpu().numpy().flatten(), num_classes)
    return val_loss/len(loader.dataset), compute_miou(conf)

# ===============================================================
# Main
# ===============================================================

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    TRAIN_IMG_DIR = "./Dataset/train_image"
    TRAIN_MASK_DIR = "./Dataset/train_mask"
    VAL_IMG_DIR   = "./Dataset/test_image"
    VAL_MASK_DIR  = "./Dataset/test_mask"
    NUM_CLASSES = 6

    train_ds = LandCoverDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, num_classes=NUM_CLASSES)
    val_ds   = LandCoverDataset(VAL_IMG_DIR, VAL_MASK_DIR, num_classes=NUM_CLASSES)
    train_dl = DataLoader(train_ds, batch_size=2, shuffle=True)
    val_dl   = DataLoader(val_ds, batch_size=2)

    models_dict = {
        "UNet": UNet(3, NUM_CLASSES).to(device),
        "SegNet": SegNet(3, NUM_CLASSES).to(device),
        "DeepLabV3": get_deeplab(NUM_CLASSES).to(device)
    }

    crit = nn.CrossEntropyLoss()
    results, losses, mious = {}, {}, {}

    for name, model in models_dict.items():
        print(f"\n🚀 Training {name}")
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        best_miou, tr_losses, val_mious = 0, [], []

        for e in range(1, 6):
            tr_loss = train_one_epoch(model, train_dl, opt, crit, device, NUM_CLASSES)
            val_loss, miou = validate(model, val_dl, crit, device, NUM_CLASSES)
            print(f"[{name}] Epoch {e}: TrainLoss={tr_loss:.4f}, ValLoss={val_loss:.4f}, mIoU={miou:.4f}")
            best_miou = max(best_miou, miou)
            tr_losses.append(tr_loss)
            val_mious.append(miou)

        torch.save(model.state_dict(), f"{name}_best.pth")
        results[name] = best_miou
        losses[name], mious[name] = tr_losses, val_mious

    # Plot metrics
    plt.figure(figsize=(10,5))
    for k,v in losses.items(): plt.plot(v,label=f"{k} Loss")
    plt.legend(); plt.title("Training Loss"); plt.show()

    plt.figure(figsize=(10,5))
    for k,v in mious.items(): plt.plot(v,label=f"{k} mIoU")
    plt.legend(); plt.title("Validation mIoU"); plt.show()

    plt.figure(figsize=(6,4))
    plt.bar(results.keys(), results.values(), color=['#66c2a5','#fc8d62','#8da0cb'])
    plt.title("Final Model Comparison (mIoU)"); plt.ylabel("mIoU"); plt.show()

    print("\n=== Final Comparison ===")
    for k,v in results.items(): print(f"{k}: {v:.4f}")
