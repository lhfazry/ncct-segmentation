#!/usr/bin/env python3
"""Stage 1: Slice-level stroke detector for two-stage NCCT segmentation.

Trains a binary classifier (stroke-present vs absent) on NCCT slices.
Outputs:
  - Detector checkpoint (.pth)
  - Per-image stroke probabilities on all splits
  - List of stroke-positive training images for Stage 2 segmentation

Usage:
  python tools/stage1_detector.py <data_root> [--work-dir] [--epochs] [--batch-size]

Two-stage pipeline:
  Stage 1 (this script): Train detector → filter training data
  Stage 2: Train segmentation model on filtered data
    python tools/train.py configs/ncct/unet_fcn_stroke_ncct.py --cfg-options \
      dataset.filter_stroke_positive=True
"""

import argparse, os, json, glob, time, warnings
import numpy as np
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, confusion_matrix, classification_report)
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train slice-level stroke detector for two-stage NCCT segmentation')
    parser.add_argument('data_root', type=str, help='Path to NCCT data root')
    parser.add_argument('--work-dir', type=str, default=None,
                        help='Output directory (default: {data_root}/detector)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50', 'efficientnet_b0'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


class NCCTDetectionDataset(Dataset):
    """NCCT slice-level binary classification dataset.

    Label: 1 if image contains any stroke pixel, 0 otherwise.
    """

    def __init__(self, data_root, split='train', transform=None):
        self.img_dir = os.path.join(data_root, split, 'images')
        self.mask_dir = os.path.join(data_root, split, 'masks')
        self.transform = transform

        self.files = sorted([
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        print(f'[{split}] Found {len(self.files)} images in {self.img_dir}')

        # Compute labels
        self.labels = []
        self.has_stroke_names = []
        self.no_stroke_names = []
        for fname in self.files:
            mask_path = os.path.join(self.mask_dir, fname)
            if os.path.exists(mask_path):
                mask = np.array(Image.open(mask_path).convert('L'))
                has_stroke = int((mask > 127).sum() > 0)
                self.labels.append(has_stroke)
                if has_stroke:
                    self.has_stroke_names.append(fname)
                else:
                    self.no_stroke_names.append(fname)
            else:
                self.labels.append(0)
                self.no_stroke_names.append(fname)

        self.labels = np.array(self.labels, dtype=np.int64)
        n_pos = self.labels.sum()
        n_neg = len(self.labels) - n_pos
        print(f'  Labels: {n_pos} positive (with stroke), {n_neg} negative')
        print(f'  Ratio: {n_neg / max(n_pos, 1):.1f}:1 (neg:pos)')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_path = os.path.join(self.img_dir, fname)
        image = Image.open(img_path).convert('L')  # grayscale
        image = image.convert('RGB')  # ResNet expects 3-channel

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label, fname

    def get_stroke_positive_files(self):
        """Return list of filenames containing stroke (for Stage 2)."""
        return self.has_stroke_names.copy()


def get_transform(train=True):
    """Get image transforms for detector training/validation."""
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=15, translate=(0.05, 0.05),
                                    scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


def create_model(backbone_name='resnet18', num_classes=2):
    """Create classifier from torchvision backbone."""
    if backbone_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights='IMAGENET1K_V1')
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes),
        )
    else:
        weights_enum = {'resnet18': models.ResNet18_Weights.IMAGENET1K_V1,
                        'resnet34': models.ResNet34_Weights.IMAGENET1K_V1,
                        'resnet50': models.ResNet50_Weights.IMAGENET1K_V1}
        model = models.__dict__[backbone_name](weights=weights_enum[backbone_name])
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    return model


def get_class_weights(labels):
    """Compute inverse-frequency weights for class-balanced loss."""
    counts = np.bincount(labels)
    weights = len(labels) / (len(counts) * counts.astype(float))
    return torch.FloatTensor(weights)


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model on a dataloader."""
    model.eval()
    all_preds, all_labels, all_probs, all_names = [], [], [], []

    for images, labels, fnames in tqdm(loader, desc='Evaluating', leave=False):
        images = images.to(device)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        preds = (probs > 0.5).astype(int)

        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())
        all_names.extend(fnames)

    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'auc_roc': roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0,
    }

    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0)
    metrics.update({'precision': prec, 'recall': rec, 'f1': f1})

    return metrics, list(zip(all_names, all_labels, all_probs, all_preds))


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    work_dir = args.work_dir or os.path.join(args.data_root, 'detector')
    os.makedirs(work_dir, exist_ok=True)
    print(f'Work dir: {work_dir}')
    print(f'Device: {args.device}')

    # ── Datasets ──
    train_dataset = NCCTDetectionDataset(args.data_root, 'train',
                                         transform=get_transform(train=True))
    val_dataset = NCCTDetectionDataset(args.data_root, 'val',
                                       transform=get_transform(train=False))
    test_dataset = NCCTDetectionDataset(args.data_root, 'test',
                                        transform=get_transform(train=False))

    # ── Class-balanced sampler ──
    labels = train_dataset.labels
    class_counts = np.bincount(labels)
    sample_weights = 1.0 / class_counts[labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2,
                            shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size * 2,
                             shuffle=False, num_workers=4, pin_memory=True)

    # ── Model ──
    model = create_model(args.backbone)
    model.to(args.device)

    class_weights = get_class_weights(labels).to(args.device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Training ──
    best_val_f1 = 0.0
    history = {'train_loss': [], 'val_metrics': []}

    print(f'\nTraining {args.backbone} detector for {args.epochs} epochs...')
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}',
                    leave=False)

        for images, labels, _ in pbar:
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        history['train_loss'].append(avg_loss)

        # Validate
        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            val_metrics, _ = evaluate(model, val_loader, args.device)
            history['val_metrics'].append({'epoch': epoch, **val_metrics})
            print(f'  Epoch {epoch:3d} | Loss: {avg_loss:.4f} | '
                  f'Val Acc: {val_metrics["accuracy"]:.4f} | '
                  f'F1: {val_metrics["f1"]:.4f} | '
                  f'Recall: {val_metrics["recall"]:.4f}')

            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                torch.save(model.state_dict(),
                           os.path.join(work_dir, 'best_detector.pth'))
                print(f'  → New best model saved (F1={best_val_f1:.4f})')

    # ── Final Evaluation ──
    print('\n' + '=' * 60)
    print('Final Evaluation')

    # Load best model
    best_ckpt = os.path.join(work_dir, 'best_detector.pth')
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=args.device))
        print(f'Loaded best model from {best_ckpt}')

    for split_name, dataset, loader in [('Train', train_dataset, train_loader),
                                         ('Val', val_dataset, val_loader),
                                         ('Test', test_dataset, test_loader)]:
        metrics, per_image = evaluate(model, loader, args.device)
        print(f'\n{split_name} Set:')
        print(f'  Accuracy:  {metrics["accuracy"]:.4f}')
        print(f'  AUC-ROC:   {metrics["auc_roc"]:.4f}')
        print(f'  Precision: {metrics["precision"]:.4f}')
        print(f'  Recall:    {metrics["recall"]:.4f}')
        print(f'  F1 Score:  {metrics["f1"]:.4f}')

        # Save per-image predictions
        out_file = os.path.join(work_dir, f'{split_name.lower()}_predictions.json')
        with open(out_file, 'w') as f:
            json.dump([{'file': name, 'label': lbl, 'prob': prob, 'pred': pred}
                       for name, lbl, prob, pred in per_image], f, indent=2)
        print(f'  Saved predictions to {out_file}')

        # Count false negatives (missed strokes)
        fns = [(name, prob) for name, lbl, prob, pred in per_image
               if lbl == 1 and pred == 0]
        if fns:
            print(f'  False negatives (missed strokes): {len(fns)}')
            for name, prob in fns[:5]:
                print(f'    {name}: prob={prob:.4f}')

    # ── Output: stroke-positive training files for Stage 2 ──
    pos_files = train_dataset.get_stroke_positive_files()
    pos_list_path = os.path.join(work_dir, 'stroke_positive_train.txt')
    with open(pos_list_path, 'w') as f:
        for fn in pos_files:
            f.write(fn + '\n')
    print(f'\nStage 2 input: {pos_list_path} ({len(pos_files)} stroke-positive files)')

    # ── Save training history ──
    with open(os.path.join(work_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2, default=str)

    print(f'\nDone. All outputs in {work_dir}')
    print(f'Use --work-dir to change output directory.')


if __name__ == '__main__':
    main()
