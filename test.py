#!/usr/bin/env python3
# ===============================================================
# 🛰️ Remote Sensing Land Cover Comparison (April vs May)
# Models: UNet, SegNet, DeepLabV3
# ===============================================================

import torch, numpy as np, matplotlib.pyplot as plt, os, csv
from torchvision import transforms
from PIL import Image
from train import UNet, SegNet, get_deeplab

# ===============================================================
# CONFIGURATION
# ===============================================================

NUM_CLASSES = 6
CLASS_LABELS = ['Water', 'Vegetation', 'Urban', 'Forest', 'Road', 'Background']
CLASS_COLORS = np.array([
    [0, 0, 255],      # Water
    [0, 255, 0],      # Vegetation
    [128, 128, 128],  # Urban
    [34, 139, 34],    # Forest
    [255, 255, 0],    # Road
    [0, 0, 0]         # Background
], dtype=np.uint8)

IMG_SIZE = (256, 256)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

APRIL_IMAGE = "./Dataset/test_image/april.jpg"
MAY_IMAGE   = "./Dataset/test_image/may.jpg"

MODEL_PATHS = {
    "UNet": "./UNet_best.pth",
    "SegNet": "./SegNet_best.pth",
    "DeepLabV3": "./DeepLabV3_best.pth"
}

# ===============================================================
# HELPER FUNCTIONS
# ===============================================================

def decode_segmap(mask):
    """Convert class index mask to RGB color mask."""
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for i in range(len(CLASS_COLORS)):
        rgb[mask == i] = CLASS_COLORS[i]
    return rgb

def compute_class_areas(pred):
    """Compute area coverage (%) for each class."""
    total_pixels = pred.size
    class_areas = [(pred == i).sum() / total_pixels * 100 for i in range(NUM_CLASSES)]
    return np.round(class_areas, 2)

def segment_image(model, image_path):
    """Run segmentation and return prediction and image."""
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor()
    ])
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(input_tensor)
        if isinstance(out, dict):
            out = out['out']
        pred = torch.argmax(out, 1).squeeze().cpu().numpy()
    return pred, img

# ===============================================================
# LOAD MODELS
# ===============================================================

def load_models():
    models = {
        "UNet": UNet(3, NUM_CLASSES),
        "SegNet": SegNet(3, NUM_CLASSES),
        "DeepLabV3": get_deeplab(NUM_CLASSES)
    }
    for name, model in models.items():
        path = MODEL_PATHS[name]
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Model weights not found for {name} at {path}")
        print(f"✅ Loading {name} from {path}")
        state_dict = torch.load(path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(DEVICE).eval()
    return models

# ===============================================================
# MAIN
# ===============================================================

print("\n🚀 Starting April vs May Comparison for All Models...\n")
models = load_models()

results = {}

# Run for April and May for all models
for name, model in models.items():
    print(f"🧠 Running {name} for April...")
    pred_april, img_april = segment_image(model, APRIL_IMAGE)
    print(f"🧠 Running {name} for May...")
    pred_may, img_may = segment_image(model, MAY_IMAGE)

    areas_april = compute_class_areas(pred_april)
    areas_may   = compute_class_areas(pred_may)

    results[name] = {
        "April": areas_april,
        "May": areas_may
    }

    # Save overlay comparison per model
    overlay_april = np.array(img_april.resize(IMG_SIZE)) * 0.5 + decode_segmap(pred_april) * 0.5
    overlay_may   = np.array(img_may.resize(IMG_SIZE)) * 0.5 + decode_segmap(pred_may) * 0.5
    overlay_april, overlay_may = overlay_april.astype(np.uint8), overlay_may.astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(img_april)
    axes[0].set_title(f"{name}: April (Original)")
    axes[1].imshow(overlay_april)
    axes[1].set_title(f"{name}: April Segmentation")
    axes[2].imshow(overlay_may)
    axes[2].set_title(f"{name}: May Segmentation")
    for ax in axes: ax.axis("off")
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, f"{name}_april_may_segmentation.png")
    plt.savefig(save_path, dpi=200)
    plt.close(fig)

# ===============================================================
# 📊 VISUALIZATION — COMBINED BAR CHART
# ===============================================================

x = np.arange(len(CLASS_LABELS))
width = 0.12

plt.figure(figsize=(12, 6))
colors = {"UNet":"#4C9AFF", "SegNet":"#FF8C42", "DeepLabV3":"#4CAF50"}

for i, (model_name, data) in enumerate(results.items()):
    plt.bar(x + (i - 1.2)*width, data["April"], width, label=f"{model_name} - April", alpha=0.8)
    plt.bar(x + (i + 0.2)*width, data["May"], width, label=f"{model_name} - May", alpha=0.8)

plt.ylabel("Area (%)")
plt.title("🌍 Land Cover Comparison (April vs May) — Across All Models")
plt.xticks(x, CLASS_LABELS, rotation=15)
plt.legend(ncol=2, fontsize=9)
plt.tight_layout()

bar_path = os.path.join(RESULTS_DIR, "comparison_all_models_april_may.png")
plt.savefig(bar_path, dpi=200)
plt.show()

# ===============================================================
# SAVE CSV
# ===============================================================

csv_path = os.path.join(RESULTS_DIR, "comparison_all_models_april_may.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "Month"] + CLASS_LABELS)
    for model_name, data in results.items():
        writer.writerow([model_name, "April"] + list(data["April"]))
        writer.writerow([model_name, "May"] + list(data["May"]))

print("\n✅ Saved results to:")
print(f"   📊 Bar chart : {bar_path}")
print(f"   📄 CSV file  : {csv_path}\n")

print("=== 📈 Per-Class Comparison Summary ===")
for model_name, data in results.items():
    print(f"\n🧩 {model_name}")
    for label, a, m in zip(CLASS_LABELS, data["April"], data["May"]):
        print(f"  {label:<12}: April = {a:.2f}% | May = {m:.2f}% | Δ = {m - a:+.2f}%")
