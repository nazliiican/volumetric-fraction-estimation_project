
import copy
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

from . import targets

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_transforms(train=True, image_size=224):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def preload_images(image_df, image_size=224):

    try:
        from tqdm import tqdm
        paths = tqdm(image_df["image_path"].unique(), desc="Preloading images")
    except Exception:  
        paths = image_df["image_path"].unique()

    small = (image_size + 32, image_size + 32)
    cache = {}
    for path in paths:
        img = Image.open(path)
        img.draft("RGB", (image_size * 2, image_size * 2))
        cache[path] = img.convert("RGB").resize(small)
    return cache


class CompositionImageDataset(Dataset):

    def __init__(self, image_df, target_by_code, transform, image_size=224,
                 image_cache=None):
        self.image_df = image_df.reset_index(drop=True)
        self.target_by_code = target_by_code
        self.transform = transform
        self.image_size = image_size
        self.image_cache = image_cache

    def __len__(self):
        return len(self.image_df)

    def _load_image(self, image_path):
        if self.image_cache is not None and image_path in self.image_cache:
            return self.image_cache[image_path]
        img = Image.open(image_path)
        img.draft("RGB", (self.image_size * 2, self.image_size * 2))
        return img.convert("RGB")

    def __getitem__(self, idx):
        row = self.image_df.iloc[idx]
        code = row["Code"]
        image_path = row["image_path"]

        image_tensor = self.transform(self._load_image(image_path))
        target = torch.tensor(self.target_by_code[code], dtype=torch.float32)
        return image_tensor, target, code, image_path


def composition_loss(pred, target, loss="kl", eps=1e-8):

    if loss == "kl":
        return F.kl_div(torch.log(pred + eps), target, reduction="batchmean")
    if loss == "mae":
        return torch.mean(torch.abs(pred - target))
    raise ValueError(f"unknown loss {loss!r} (use 'kl' or 'mae')")


@torch.no_grad()
def _evaluate_loss(model, loader, device, loss, eps):
    model.eval()
    losses, maes = [], []
    for image_tensor, target, _, _ in loader:
        image_tensor, target = image_tensor.to(device), target.to(device)
        pred = model(image_tensor)
        losses.append(composition_loss(pred, target, loss, eps).item())
        maes.append(torch.mean(torch.abs(pred - target)).item())
    return float(np.mean(losses)), float(np.mean(maes))


def train_composition_model(model, train_loader, val_loader, device, epochs=30,
                            lr=1e-3, patience=7, loss="kl", save_path=None,
                            verbose=True):
    
    model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr)
    eps = 1e-8

    history = []
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses, batch_maes = [], []
        for image_tensor, target, _, _ in train_loader:
            image_tensor, target = image_tensor.to(device), target.to(device)
            optimizer.zero_grad()
            pred = model(image_tensor)
            batch_loss = composition_loss(pred, target, loss, eps)
            batch_loss.backward()
            optimizer.step()
            batch_losses.append(batch_loss.item())
            batch_maes.append(torch.mean(torch.abs(pred - target)).item())

        train_loss = float(np.mean(batch_losses))
        train_mae = float(np.mean(batch_maes))
        val_loss, val_mae = _evaluate_loss(model, val_loader, device, loss, eps)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "train_mae": train_mae,
                        "val_mae": val_mae})

        if verbose:
            print(f"epoch {epoch:2d} | train_loss {train_loss:.4f} "
                  f"val_loss {val_loss:.4f} | train_mae {train_mae:.4f} "
                  f"val_mae {val_mae:.4f}")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"early stopping at epoch {epoch} "
                          f"(no val improvement for {patience} epochs)")
                break

    model.load_state_dict(best_state)
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model, pd.DataFrame(history)


@torch.no_grad()
def predict_composition_model(model, loader, device, materials, split_name=None):
    
    model.eval()
    codes, paths, preds, trues = [], [], [], []
    for image_tensor, target, code, image_path in loader:
        pred = model(image_tensor.to(device)).cpu().numpy()
        preds.append(pred)
        trues.append(target.numpy())
        codes.extend(list(code))
        paths.extend(list(image_path))

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    df = pd.DataFrame({"Code": codes, "image_path": paths})
    for j, material in enumerate(materials):
        df[f"true_{material}"] = trues[:, j]
        df[f"pred_{material}"] = preds[:, j]
    if split_name is not None:
        df["split"] = split_name
    return df


def renormalize_composition(matrix, eps=1e-12):
    """Clip negatives to 0 and renormalize each row to sum to 1."""
    m = np.clip(np.asarray(matrix, dtype=float), 0.0, None)
    row_sums = m.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return m / row_sums


class MultitaskImageDataset(Dataset):
    

    def __init__(self, image_df, composition_by_code, presence_by_code, transform,
                 image_size=224, image_cache=None):
        self.image_df = image_df.reset_index(drop=True)
        self.composition_by_code = composition_by_code
        self.presence_by_code = presence_by_code
        self.transform = transform
        self.image_size = image_size
        self.image_cache = image_cache

    def __len__(self):
        return len(self.image_df)

    def _load_image(self, image_path):
        if self.image_cache is not None and image_path in self.image_cache:
            return self.image_cache[image_path]
        img = Image.open(image_path)
        img.draft("RGB", (self.image_size * 2, self.image_size * 2))
        return img.convert("RGB")

    def __getitem__(self, idx):
        row = self.image_df.iloc[idx]
        code = row["Code"]
        image_path = row["image_path"]
        image_tensor = self.transform(self._load_image(image_path))
        composition = torch.tensor(self.composition_by_code[code], dtype=torch.float32)
        presence = torch.tensor(self.presence_by_code[code], dtype=torch.float32)
        return image_tensor, composition, presence, code, image_path


def multitask_loss(outputs, target_volume, target_presence,
                   alpha=1.0, beta=1.0, gamma=0.01, eps=1e-8):
    
    presence_loss = F.binary_cross_entropy_with_logits(
        outputs["presence_logits"], target_presence)
    fused = outputs["fused_composition"]
    composition_loss = F.kl_div(torch.log(fused + eps), target_volume,
                                reduction="batchmean")
    sparsity_loss = outputs["presence_prob"].mean()

    total = alpha * presence_loss + beta * composition_loss + gamma * sparsity_loss
    parts = {"presence": float(presence_loss.item()),
             "composition": float(composition_loss.item()),
             "sparsity": float(sparsity_loss.item())}
    return total, parts


@torch.no_grad()
def _evaluate_multitask(model, loader, device, alpha, beta, gamma, eps):
    model.eval()
    losses, maes = [], []
    tp = fp = fn = 0
    for image_tensor, composition, presence, _, _ in loader:
        image_tensor = image_tensor.to(device)
        composition = composition.to(device)
        presence = presence.to(device)
        out = model(image_tensor)
        loss, _ = multitask_loss(out, composition, presence, alpha, beta, gamma, eps)
        losses.append(loss.item())
        maes.append(torch.mean(torch.abs(out["fused_composition"] - composition)).item())
        pred_present = out["presence_prob"] > 0.5
        true_present = presence > 0.5
        tp += int((true_present & pred_present).sum())
        fp += int((~true_present & pred_present).sum())
        fn += int((true_present & ~pred_present).sum())
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    return float(np.mean(losses)), float(np.mean(maes)), float(f1)


def train_multitask_model(model, train_loader, val_loader, device, epochs=30,
                          lr=1e-3, patience=7, alpha=1.0, beta=1.0, gamma=0.01,
                          save_path=None, verbose=True):
    
    model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr)
    eps = 1e-8

    history = []
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses, batch_maes = [], []
        for image_tensor, composition, presence, _, _ in train_loader:
            image_tensor = image_tensor.to(device)
            composition = composition.to(device)
            presence = presence.to(device)
            optimizer.zero_grad()
            out = model(image_tensor)
            loss, _ = multitask_loss(out, composition, presence, alpha, beta, gamma, eps)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
            batch_maes.append(
                torch.mean(torch.abs(out["fused_composition"] - composition)).item())

        train_loss = float(np.mean(batch_losses))
        train_mae = float(np.mean(batch_maes))
        val_loss, val_mae, val_f1 = _evaluate_multitask(
            model, val_loader, device, alpha, beta, gamma, eps)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "train_mae": train_mae, "val_mae": val_mae,
                        "val_presence_f1": val_f1})
        if verbose:
            print(f"epoch {epoch:2d} | train_loss {train_loss:.4f} "
                  f"val_loss {val_loss:.4f} | val_mae {val_mae:.4f} "
                  f"val_presence_f1 {val_f1:.4f}")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"early stopping at epoch {epoch} "
                          f"(no val improvement for {patience} epochs)")
                break

    model.load_state_dict(best_state)
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model, pd.DataFrame(history)


@torch.no_grad()
def predict_multitask_model(model, loader, device, materials, density_vector,
                            split_name=None):
    
    model.eval()
    codes, paths = [], []
    fused, softmax_comp, presence_prob, true_vol, true_presence = [], [], [], [], []
    for image_tensor, composition, presence, code, image_path in loader:
        out = model(image_tensor.to(device))
        fused.append(out["fused_composition"].cpu().numpy())
        softmax_comp.append(out["composition"].cpu().numpy())
        presence_prob.append(out["presence_prob"].cpu().numpy())
        true_vol.append(composition.numpy())
        true_presence.append(presence.numpy())
        codes.extend(list(code))
        paths.extend(list(image_path))

    fused = np.concatenate(fused)
    softmax_comp = np.concatenate(softmax_comp)
    presence_prob = np.concatenate(presence_prob)
    true_vol = np.concatenate(true_vol)
    true_presence = np.concatenate(true_presence)

    density = np.asarray(density_vector, dtype=float)
    fused_mass = targets.volume_to_mass(fused, density)
    true_mass = targets.volume_to_mass(true_vol, density)

    df = pd.DataFrame({"Code": codes, "image_path": paths})
    if split_name is not None:
        df["split"] = split_name
    for j, m in enumerate(materials):
        df[f"true_{m}"] = true_mass[:, j]
        df[f"pred_{m}"] = fused_mass[:, j]
        df[f"true_vol_{m}"] = true_vol[:, j]
        df[f"pred_vol_{m}"] = fused[:, j]
        df[f"true_presence_{m}"] = true_presence[:, j]
        df[f"pred_presence_{m}"] = presence_prob[:, j]
        df[f"softmax_vol_{m}"] = softmax_comp[:, j]
    return df
