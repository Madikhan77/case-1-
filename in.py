import os
import json
import random
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

from sklearn.metrics import precision_recall_fscore_support, balanced_accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm

# --- Paths ---
ROOT = Path("dataset")
TRAIN_IMAGES_DIR = ROOT / "image" / "image"
VAL_IMAGES_DIR = ROOT / "validation" / "validation"

TRAIN_ANN_VIA = Path("0Train_via_annos.json")
VAL_ANN_VIA = Path("0Val_via_annos.json")

# --- Classes (8) ---
FINAL_CLASSES = [
    "dent",
    "scratch",
    "broken glass",
    "lost parts",
    "punctured",
    "torn",
    "broken lights",
    "non-damage",
]
NUM_CLASSES = len(FINAL_CLASSES)


STRATEGY = "weighted"

# --- Train ---
IMG_SIZE = 448
BATCH_SIZE = 8
EPOCHS = 7
WARMUP_EPOCHS = 3
BASE_LR = 3e-4
FINETUNE_LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "resnet50"

CHECKPOINT_BEST = "vehide_cls8_resnet50_best.pt"
CHECKPOINT_LAST = "vehide_cls8_resnet50_last.pt"
HISTORY_PATH = "train_history.json"

def _normalize_token(x: str) -> str:
    return str(x).strip().lower().replace(" ", "_")

# ---- Прямое сопоставление сырых меток к финальным 8 классам ----

RAW2FINAL = {
    # лёгкие/косметические
    "scratch": "scratch",
    "rach": "scratch",
    "paint": "scratch",
    "paint_damage": "scratch",
    "tray_son": "scratch",

    # «вмятины/корпус»
    "mop_lom": "dent",
    "body_damage": "dent",
    "door_dent": "dent",
    "bumper_dent": "dent",
    "thung": "dent",  # пробоина/вмятина → чаще как dent

    # потеря деталей
    "mat_bo_phan": "lost parts",

    # прокол/разрыв
    "punctured": "punctured",
    "torn": "torn",

    # стекло/оптика
    "broken_glass": "broken glass",
    "glass_shatter": "broken glass",
    "vo_kinh": "broken glass",

    "broken_lights": "broken lights",
    "headlight_damage": "broken lights",
    "broken_lights_head": "broken lights",
    "tail_lamp_broken": "broken lights",
    "be_den": "broken lights",
}

# какие ключи выцеплять из region / region_attributes
CANDIDATE_ATTR_KEYS = [
    "label", "labels", "class", "classes", "category", "categories",
    "damage_type", "damage", "type", "name", "tag", "tags"
]
CONFIDENCE_KEYS = ["confidence", "score", "prob", "probability"]

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def extract_raw_label_from_region(region: Dict[str, Any]) -> Optional[str]:
    """Ставит твои варианты: и прямые поля, и в region_attributes, и dict/list форматы."""
    # прямые поля
    for k in ("class", "label", "type", "name", "category"):
        v = region.get(k)
        if isinstance(v, str) and v.strip():
            return _normalize_token(v)

    # в region_attributes
    attrs = region.get("region_attributes") or {}
    if isinstance(attrs, dict) and attrs:
        for k in CANDIDATE_ATTR_KEYS:
            v = attrs.get(k)
            if isinstance(v, str) and v.strip():
                return _normalize_token(v)

        for k in CANDIDATE_ATTR_KEYS:
            v = attrs.get(k)
            if isinstance(v, dict):
                for kk, vv in v.items():
                    if isinstance(vv, (bool, int, float)) and bool(vv):
                        return _normalize_token(kk)
                    if isinstance(vv, str) and vv.strip():
                        return _normalize_token(vv)

        for k in CANDIDATE_ATTR_KEYS:
            v = attrs.get(k)
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, str) and item.strip():
                        return _normalize_token(item)
    return None

def extract_confidence_from_region(region: Dict[str, Any], default_if_missing: float = 1.0) -> float:
    attrs = region.get("region_attributes", {}) or {}
    for key in CONFIDENCE_KEYS:
        if key in attrs:
            try:
                return float(attrs[key])
            except:
                pass
    return default_if_missing

def raw_to_final(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    key = _normalize_token(raw)
    # прямое попадание
    if key in RAW2FINAL:
        return RAW2FINAL[key]

    # эвристики на всякий случай
    if "scratch" in key or "rach" in key or "paint" in key or "tray_son" in key:
        return "scratch"
    if "glass" in key or "vo_kinh" in key:
        return "broken glass"
    if any(t in key for t in ["dent", "lom", "body", "thung"]):
        return "dent"
    if "mat_bo_phan" in key:
        return "lost parts"
    if "punctur" in key:
        return "punctured"
    if "torn" in key:
        return "torn"
    if any(t in key for t in ["light", "lamp", "den"]):
        return "broken lights"
    return None

def load_via_items(via_json_path: Path) -> List[Tuple[str, List[Dict[str, Any]]]]: 
    with open(via_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "via_img_metadata" in data:
        metas = list(data["via_img_metadata"].values())
    elif isinstance(data, list):
        metas = data
    else:
        metas = list(data.values())

    items = []
    for m in metas:
        fn = m.get("filename") or m.get("file_name") or m.get("name")
        regs = m.get("regions", []) or []
        if fn:
            items.append((fn, regs))
    return items

def aggregate_regions_to_final_label(regions: List[Dict[str, Any]], strategy: str = STRATEGY) -> Optional[str]:

    if not regions:
        return "non-damage"

    scores = {c: 0.0 for c in FINAL_CLASSES}
    found_any = False

    for r in regions:
        raw = extract_raw_label_from_region(r)
        final = raw_to_final(raw)
        if final is None:
            continue
        conf = extract_confidence_from_region(r, default_if_missing=1.0)
        scores[final] += float(conf)
        found_any = True

    if not found_any:
        return "non-damage"

    # не даём «non-damage» побеждать, если были найденные метки
    scores_no_nd = {k: v for k, v in scores.items() if k != "non-damage"}
    best = max(scores_no_nd.items(), key=lambda kv: kv[1])[0]
    return best

def count_by_class(pairs):
    return dict(Counter([FINAL_CLASSES[y] for _, y in pairs]))

def build_samples_from_via(images_dir, via_json_path, strategy=STRATEGY, drop_unlabeled=True):
    items = load_via_items(via_json_path)
    samples = []
    for fn, regs in items:
        lbl = aggregate_regions_to_final_label(regs, strategy)

        # если вообще не смогли агрегировать (маловероятно теперь) — трактуем как non-damage
        if lbl is None:
            lbl = "non-damage"

        if lbl not in FINAL_CLASSES:
            continue

        path = images_dir / fn
        if not path.exists():
            continue

        samples.append((path, FINAL_CLASSES.index(lbl)))

    print(f"[build] {via_json_path.name}: {len(samples)} samples | dist={count_by_class(samples)}")
    return samples

class VehicleDamageDataset(Dataset):
    def __init__(self, pairs, transform=None):
        self.samples = pairs
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, lbl = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
        if self.transform:
            img = self.transform(img)
        return img, lbl

def get_transforms():
    train = transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return train, val

class VehicleDamageModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, model_name=MODEL_NAME):
        super().__init__()
        self.model_name = model_name
        if model_name == "resnet50":
            self.backbone = models.resnet50(weights="IMAGENET1K_V2")
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        elif model_name == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(weights="IMAGENET1K_V1")
            self.backbone.classifier[1] = nn.Linear(self.backbone.classifier[1].in_features, num_classes)
        else:
            raise ValueError

    def forward(self, x):
        return self.backbone(x)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        if self.model_name == "resnet50":
            for p in self.backbone.fc.parameters():
                p.requires_grad = True

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

def train_epoch(model, loader, crit, opt, dev):
    model.train(); loss=0; cor=0; tot=0
    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad()
        out = model(x)
        l = crit(out, y)
        l.backward()
        opt.step()
        loss += l.item()
        cor += out.argmax(1).eq(y).sum().item()
        tot += y.size(0)
    return loss/len(loader), 100*cor/max(1, tot)

def validate_epoch(model, loader, crit, dev):
    model.eval(); loss=0; cor=0; tot=0; preds=[]; targs=[]
    with torch.no_grad():
        for x, y in tqdm(loader, desc="valid", leave=False):
            x, y = x.to(dev), y.to(dev)
            out = model(x)
            l = crit(out, y)
            loss += l.item()
            p = out.argmax(1)
            cor += p.eq(y).sum().item()
            tot += y.size(0)
            preds += p.cpu().tolist()
            targs += y.cpu().tolist()
    return loss/len(loader), 100*cor/max(1, tot), preds, targs

def plot_confusion(y_pred, y_true, classes, normalize=None, fname="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    plt.figure(figsize=(7.5, 6))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d",
                xticklabels=classes, yticklabels=classes, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

def main():
    print(f"Using device: {DEVICE}")
    set_seed(SEED)

    train_tf, val_tf = get_transforms()

    # Build samples
    train_pairs = build_samples_from_via(TRAIN_IMAGES_DIR, TRAIN_ANN_VIA, strategy=STRATEGY, drop_unlabeled=True)
    val_pairs   = build_samples_from_via(VAL_IMAGES_DIR,   VAL_ANN_VIA,   strategy=STRATEGY, drop_unlabeled=True)

    if len(train_pairs) == 0:
        raise RuntimeError("No training samples! Check paths/annotations/strategy.")
    if len(val_pairs) == 0:
        raise RuntimeError("No validation samples! Check paths/annotations/strategy.")

    train_ds = VehicleDamageDataset(train_pairs, train_tf)
    val_ds   = VehicleDamageDataset(val_pairs,   val_tf)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # class weights (simple inverse freq; можно заменить на посложнее при желании)
    counts = Counter([y for _, y in train_pairs])
    weights = torch.tensor([1.0/max(1, counts.get(i, 1)) for i in range(NUM_CLASSES)], dtype=torch.float32, device=DEVICE)
    weights = weights * (NUM_CLASSES / weights.sum())
    print("class weights:", weights.cpu().numpy().round(3).tolist())

    model = VehicleDamageModel(num_classes=NUM_CLASSES, model_name=MODEL_NAME).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # Phase 1: warmup
    model.freeze_backbone()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=BASE_LR, weight_decay=WEIGHT_DECAY)

    best_val = 0.0
    print("\n=== PHASE 1: WARMUP ===")
    for e in tqdm(range(WARMUP_EPOCHS), desc="warmup-epochs"):
        tr_loss, tr_acc = train_epoch(model, train_dl, criterion, optimizer, DEVICE)
        va_loss, va_acc, va_preds, va_targs = validate_epoch(model, val_dl, criterion, DEVICE)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
            va_targs, va_preds, labels=list(range(NUM_CLASSES)),
            average="macro", zero_division=0
        )
        bal_acc = balanced_accuracy_score(va_targs, va_preds)

        print(f"Warmup {e+1}/{WARMUP_EPOCHS}: "
              f"train {tr_acc:.2f}% | val {va_acc:.2f}% | "
              f"macroF1 {f1_m:.3f} | macroP {prec_m:.3f} | macroR {rec_m:.3f} | bAcc {bal_acc:.3f}")

        if va_acc > best_val:
            best_val = va_acc
            torch.save({
                "epoch": e + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val,
                "config": {
                    "STRATEGY": STRATEGY,
                    "MODEL_NAME": MODEL_NAME,
                    "IMG_SIZE": IMG_SIZE,
                    "BATCH_SIZE": BATCH_SIZE,
                    "EPOCHS": EPOCHS,
                    "WARMUP_EPOCHS": WARMUP_EPOCHS,
                    "FINAL_CLASSES": FINAL_CLASSES,
                },
            }, CHECKPOINT_BEST)
            print(f" -> saved best ({best_val:.2f}%)")

    # Phase 2: fine-tune
    print("\n=== PHASE 2: FINE-TUNE ===")
    model.unfreeze_backbone()
    optimizer = optim.AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=WEIGHT_DECAY)

    for e in tqdm(range(WARMUP_EPOCHS, EPOCHS), desc="finetune-epochs"):
        tr_loss, tr_acc = train_epoch(model, train_dl, criterion, optimizer, DEVICE)
        va_loss, va_acc, va_preds, va_targs = validate_epoch(model, val_dl, criterion, DEVICE)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
            va_targs, va_preds, labels=list(range(NUM_CLASSES)),
            average="macro", zero_division=0
        )
        bal_acc = balanced_accuracy_score(va_targs, va_preds)

        print(f"Epoch {e+1}/{EPOCHS}: "
              f"train {tr_acc:.2f}% | val {va_acc:.2f}% | "
              f"macroF1 {f1_m:.3f} | macroP {prec_m:.3f} | macroR {rec_m:.3f} | bAcc {bal_acc:.3f}")

        if va_acc > best_val:
            best_val = va_acc
            torch.save({
                "epoch": e + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val,
                "config": {
                    "STRATEGY": STRATEGY,
                    "MODEL_NAME": MODEL_NAME,
                    "IMG_SIZE": IMG_SIZE,
                    "BATCH_SIZE": BATCH_SIZE,
                    "EPOCHS": EPOCHS,
                    "WARMUP_EPOCHS": WARMUP_EPOCHS,
                    "FINAL_CLASSES": FINAL_CLASSES,
                },
            }, CHECKPOINT_BEST)
            print(f" -> saved best ({best_val:.2f}%)")

    print(f"\nBest validation accuracy: {best_val:.2f}%")
    torch.save({
        "epoch": EPOCHS,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_acc": best_val,
    }, CHECKPOINT_LAST)

    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"Saved checkpoints: best={CHECKPOINT_BEST}, last={CHECKPOINT_LAST}")
    print(f"Saved history: {HISTORY_PATH}")

    # Final eval with best
    model.load_state_dict(torch.load(CHECKPOINT_BEST, map_location=DEVICE)["model_state_dict"])
    _, _, preds, targs = validate_epoch(model, val_dl, criterion, DEVICE)

    print("\nClassification report:")
    print(classification_report(targs, preds, target_names=FINAL_CLASSES, digits=4))
    plot_confusion(preds, targs, FINAL_CLASSES)
    print("Saved: confusion_matrix.png, and best checkpoint:", CHECKPOINT_BEST)

if __name__ == "__main__":
    main()
