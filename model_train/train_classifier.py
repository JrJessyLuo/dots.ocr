#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, DebertaV2Model
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
MODEL_NAME = "microsoft/mdeberta-v3-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_LLMS = 3   # number of LLMs (llm_id in [0, ..., NUM_LLMS-1])


# ---------------------------------------------------------------------
# 1. Model (Accuracy Only)
# ---------------------------------------------------------------------
class AccuracyRouter(nn.Module):
    def __init__(self, backbone_name: str, num_llms: int, hidden_dim: int = 768):
        super().__init__()
        self.backbone = DebertaV2Model.from_pretrained(backbone_name)
        self.llm_embedding = nn.Embedding(num_llms, hidden_dim)

        # Accuracy head only
        self.acc_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, input_ids, attention_mask, llm_ids):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        query_emb = outputs.last_hidden_state[:, 0, :]      # CLS representation
        llm_emb = self.llm_embedding(llm_ids)
        combined = torch.cat([query_emb, llm_emb], dim=1)
        logits = self.acc_head(combined).squeeze(-1)
        return logits


# ---------------------------------------------------------------------
# 2. Dataset (Accuracy Only)
# ---------------------------------------------------------------------
class RouterDataset(Dataset):
    """
    Each record in JSON should contain:
      - 'question' : str
      - 'llm_id'   : int in [0, NUM_LLMS-1]
      - 'accuracy' : 0/1 (float or int)
    """
    def __init__(self, data_path, tokenizer, max_len: int = 512):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item["question"],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)       # (L,)
        attention_mask = encoding["attention_mask"].squeeze(0)

        llm_id = torch.tensor(item["llm_id"], dtype=torch.long)
        acc = torch.tensor(item["accuracy"], dtype=torch.float)

        return input_ids, attention_mask, llm_id, acc


# ---------------------------------------------------------------------
# 3. Evaluation & Training
# ---------------------------------------------------------------------
def evaluate(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, mask, llm_ids, labels = [b.to(DEVICE) for b in batch]
            logits = model(input_ids, mask, llm_ids)
            preds = (torch.sigmoid(logits) > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return accuracy_score(all_labels, all_preds)


def train_loop(model, train_loader, val_loader, optimizer, epochs: int, task_name: str):
    criterion = nn.BCEWithLogitsLoss()
    best_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids, mask, llm_ids, labels = [b.to(DEVICE) for b in batch]

            optimizer.zero_grad()
            logits = model(input_ids, mask, llm_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        val_acc = evaluate(model, val_loader)
        avg_loss = total_loss / max(len(train_loader), 1)
        print(f"[{task_name}] Ep {epoch:02d}: Loss {avg_loss:.4f} | Val Acc {val_acc:.2%}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_acc


# ---------------------------------------------------------------------
# 4. Runners for baseline / proposed
# ---------------------------------------------------------------------
def run_baseline(tokenizer, dl_train, dl_eval, epochs: int = 20):
    print("\n--- Baseline (Gold only) ---")
    model = AccuracyRouter(MODEL_NAME, NUM_LLMS).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    best_acc = train_loop(model, dl_train, dl_eval, optimizer, epochs, task_name="Baseline")
    return best_acc


def run_proposed(tokenizer, dl_synth, dl_train, dl_eval, pre_epochs: int = 5, ft_epochs: int = 20):
    print("\n--- Proposed (Synth Pretrain → Gold Finetune) ---")
    model = AccuracyRouter(MODEL_NAME, NUM_LLMS).to(DEVICE)

    # Phase 1: Pretrain on synthetic data
    opt_pre = torch.optim.AdamW(model.parameters(), lr=2e-5)
    _ = train_loop(model, dl_synth, dl_eval, opt_pre, pre_epochs, task_name="Phase1-Pretrain")

    # Phase 2: Discriminative LR fine-tuning on gold data
    params = [
        {"params": model.backbone.parameters(),      "lr": 5e-6},
        {"params": model.acc_head.parameters(),      "lr": 1e-3},
        {"params": model.llm_embedding.parameters(), "lr": 1e-3},
    ]
    opt_ft = torch.optim.AdamW(params)
    best_acc = train_loop(model, dl_train, dl_eval, opt_ft, ft_epochs, task_name="Phase2-Finetune")

    return best_acc


# ---------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Accuracy Router: Baseline vs Proposed")
    parser.add_argument("--mode", choices=["baseline", "proposed", "both"],
                        default="both", help="Which training mode to run")
    parser.add_argument("--synthetic_path", default="synthetic_data.json")
    parser.add_argument("--train_path", default="train_gold.json")
    parser.add_argument("--eval_path", default="eval_gold.json")
    parser.add_argument("--batch_synth", type=int, default=8)
    parser.add_argument("--batch_train", type=int, default=8)
    parser.add_argument("--batch_eval", type=int, default=8)
    parser.add_argument("--epochs_baseline", type=int, default=20)
    parser.add_argument("--epochs_pretrain", type=int, default=5)
    parser.add_argument("--epochs_finetune", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # DataLoaders
    dl_synth = DataLoader(
        RouterDataset(args.synthetic_path, tokenizer),
        batch_size=args.batch_synth,
        shuffle=True,
    )
    dl_train = DataLoader(
        RouterDataset(args.train_path, tokenizer),
        batch_size=args.batch_train,
        shuffle=True,
    )
    dl_eval = DataLoader(
        RouterDataset(args.eval_path, tokenizer),
        batch_size=args.batch_eval,
        shuffle=False,
    )

    print("\n=== Training Accuracy Router ===")

    acc_base = None
    acc_prop = None

    if args.mode in ("baseline", "both"):
        acc_base = run_baseline(tokenizer, dl_train, dl_eval, epochs=args.epochs_baseline)

    if args.mode in ("proposed", "both"):
        # if you ran baseline first and worry about memory, you can explicitly clear it:
        torch.cuda.empty_cache()
        acc_prop = run_proposed(
            tokenizer,
            dl_synth,
            dl_train,
            dl_eval,
            pre_epochs=args.epochs_pretrain,
            ft_epochs=args.epochs_finetune,
        )

    # Summary
    if args.mode == "both" and acc_base is not None and acc_prop is not None:
        print(f"\nFinal Result: Baseline = {acc_base:.2%} | Proposed = {acc_prop:.2%}")
        print(f"Improvement: {(acc_prop - acc_base) * 100:.2f} points")
    elif args.mode == "baseline" and acc_base is not None:
        print(f"\nFinal Baseline Accuracy: {acc_base:.2%}")
    elif args.mode == "proposed" and acc_prop is not None:
        print(f"\nFinal Proposed Accuracy: {acc_prop:.2%}")
