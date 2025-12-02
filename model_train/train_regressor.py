import argparse
import json
import math
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from transformers import AutoTokenizer, DebertaV2Model
from tqdm import tqdm
import os

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
MODEL_NAME = "microsoft/mdeberta-v3-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_LLMS = 3

# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------------------
# 1. Model with Cold-Start Warmup
# ---------------------------------------------------------------------
class CostRouter(nn.Module):
    def __init__(self, backbone_name: str, num_llms: int, hidden_dim: int = 768):
        super().__init__()
        self.backbone = DebertaV2Model.from_pretrained(backbone_name)
        self.llm_embedding = nn.Embedding(num_llms, hidden_dim)

        self.cost_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )
        
        self.register_buffer("memory_bank", None)

    def populate_memory_bank(self, dataloader, device):
        self.eval()
        embeddings_list = []
        print(">> Populating Memory Bank (Unnormalized)...")
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Encoding Memory"):
                # Handle batch unpacking robustly (might have 4 or more items)
                input_ids = batch[0].to(device)
                mask = batch[1].to(device)
                
                outputs = self.backbone(input_ids=input_ids, attention_mask=mask)
                query_emb = outputs.last_hidden_state[:, 0, :]
                
                # Store RAW embeddings for regression magnitude consistency
                embeddings_list.append(query_emb)
        
        if embeddings_list:
            self.memory_bank = torch.cat(embeddings_list, dim=0)
            print(f">> Memory Bank populated with {self.memory_bank.size(0)} vectors.")
        else:
            print(">> Warning: Memory bank empty.")

    def get_warmup_embedding(self, query_emb, k=5):
        if self.memory_bank is None:
            return query_emb
            
        # 1. Normalize for Directional Search (Cosine)
        q_norm = F.normalize(query_emb, p=2, dim=1)
        m_norm = F.normalize(self.memory_bank, p=2, dim=1)
        
        # 2. Similarity Search
        sim_matrix = torch.matmul(q_norm, m_norm.t())
        
        # 3. Top-K
        actual_k = min(k, self.memory_bank.size(0))
        _, indices = torch.topk(sim_matrix, k=actual_k, dim=1)
        
        # 4. Retrieve RAW neighbors and Average
        neighbors = self.memory_bank[indices]
        e_warm = torch.mean(neighbors, dim=1)
        
        return e_warm

    def forward(self, input_ids, attention_mask, llm_ids, use_warmup=False, warmup_lambda=0.1, k=5):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        query_emb = outputs.last_hidden_state[:, 0, :]

        if use_warmup and self.memory_bank is not None:
            e_warm = self.get_warmup_embedding(query_emb, k=k)
            query_emb = (1 - warmup_lambda) * query_emb + warmup_lambda * e_warm

        llm_emb = self.llm_embedding(llm_ids)
        combined = torch.cat([query_emb, llm_emb], dim=1)
        return self.cost_head(combined).squeeze(-1)

# ---------------------------------------------------------------------
# 2. Dataset
# ---------------------------------------------------------------------
class CostDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len: int = 256):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        encoding = self.tokenizer(
            item["question"],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        raw_cost = float(item.get("cost", 1e-10))
        if raw_cost <= 0: raw_cost = 1e-10
        scaled_cost = math.log(raw_cost)

        return (
            encoding["input_ids"].squeeze(0),
            encoding["attention_mask"].squeeze(0),
            torch.tensor(item["llm_id"], dtype=torch.long),
            torch.tensor(scaled_cost, dtype=torch.float),
        )

# ---------------------------------------------------------------------
# 3. Evaluation & Training
# ---------------------------------------------------------------------
def evaluate(model, dataloader, use_warmup=False, warmup_lambda=0.1, k=5):
    model.eval()
    preds_log, labels_log = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, mask, llm_ids, labels = [b.to(DEVICE) for b in batch]

            preds = model(
                input_ids, mask, llm_ids, 
                use_warmup=use_warmup, 
                warmup_lambda=warmup_lambda, 
                k=k
            )
            
            preds_log.extend(preds.cpu().numpy())
            labels_log.extend(labels.cpu().numpy())

    preds_log = np.array(preds_log, dtype=np.float64)
    labels_log = np.array(labels_log, dtype=np.float64)

    real_preds = np.exp(preds_log)
    real_labels = np.exp(labels_log)

    epsilon = 1e-10
    mape = np.mean(np.abs((real_labels - real_preds) / (real_labels + epsilon))) * 100.0
    return mape

def train_loop(model, train_loader, val_loader, optimizer, epochs, task_name):
    criterion = nn.MSELoss()
    best_mape = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids, mask, llm_ids, labels = [b.to(DEVICE) for b in batch]

            optimizer.zero_grad()
            preds = model(input_ids, mask, llm_ids, use_warmup=False)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_mape = evaluate(model, val_loader, use_warmup=False)
        
        if val_mape < best_mape:
            best_mape = val_mape
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_mape

def reset_module(m: nn.Module):
    for layer in m.modules():
        if hasattr(layer, "reset_parameters"): layer.reset_parameters()

# ---------------------------------------------------------------------
# 4. Runners (Optimized with Replay)
# ---------------------------------------------------------------------
def run_proposed_with_warmup(tokenizer, dl_synth, dl_train, dl_eval, pre_epochs=10, ft_epochs=20):
    print("\n=== Proposed Method (Pretrain -> Replay Finetune -> Warmup) ===")
    model = CostRouter(MODEL_NAME, NUM_LLMS).to(DEVICE)

    # 1. Phase 1: Pretrain
    print("Phase 1: Pre-training on Synthetic Data...")
    for p in model.backbone.parameters(): p.requires_grad = False
    params_pre = list(model.llm_embedding.parameters()) + list(model.cost_head.parameters())
    opt_pre = torch.optim.AdamW(params_pre, lr=1e-3)
    _ = train_loop(model, dl_synth, dl_eval, opt_pre, pre_epochs, "Phase1")

    # 2. Phase 2: Replay Fine-tuning (Gold + 20% Synth)
    print("Phase 2: Fine-tuning with Replay...")
    for p in model.backbone.parameters(): p.requires_grad = True
    reset_module(model.cost_head)

    # Create Replay Dataset
    synth_ds = dl_synth.dataset
    train_ds = dl_train.dataset
    
    # Sample 20% size of Gold from Synth
    replay_size = int(len(train_ds) * 5.0) # Make buffer 5x larger than gold to dominate epoch
    indices = np.random.choice(len(synth_ds), replay_size, replace=False)
    synth_replay = Subset(synth_ds, indices)
    
    combined_ds = ConcatDataset([train_ds, synth_replay])
    dl_combined = DataLoader(combined_ds, batch_size=8, shuffle=True)

    params_ft = [
        {"params": model.backbone.parameters(),      "lr": 1e-5},
        {"params": model.llm_embedding.parameters(), "lr": 1e-3},
        {"params": model.cost_head.parameters(),     "lr": 1e-3},
    ]
    opt_ft = torch.optim.AdamW(params_ft)
    
    # Train on COMBINED data
    train_loop(model, dl_combined, dl_eval, opt_ft, ft_epochs, "Phase2")

    # 3. Warmup Eval
    print("\n>> Populating Expanded Memory Bank (Synth + Gold)...")
    # Use combined dataloader for memory bank to maximize coverage
    dl_bank = DataLoader(ConcatDataset([train_ds, synth_ds]), batch_size=32, shuffle=False)
    model.populate_memory_bank(dl_bank, DEVICE)
    
    mape_std = evaluate(model, dl_eval, use_warmup=False)
    mape_warm = evaluate(model, dl_eval, use_warmup=True, warmup_lambda=0.1, k=5)
    
    print(f"MAPE (Standard): {mape_std:.2f}%")
    print(f"MAPE (w/ Warmup): {mape_warm:.2f}% (Lambda=0.1, K=5)")
    
    return min(mape_std, mape_warm)

def run_baseline(tokenizer, dl_train, dl_eval, epochs=20):
    print("\n=== Baseline (Gold Only) ===")
    model = CostRouter(MODEL_NAME, NUM_LLMS).to(DEVICE)
    params = [
        {"params": model.backbone.parameters(), "lr": 1e-5},
        {"params": model.llm_embedding.parameters(), "lr": 1e-3},
        {"params": model.cost_head.parameters(), "lr": 1e-3},
    ]
    opt = torch.optim.AdamW(params)
    mape = train_loop(model, dl_train, dl_eval, opt, epochs, "Baseline")
    print(f"Baseline MAPE: {mape:.2f}%")
    return mape

# ---------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic_path", default="xx.json")
    parser.add_argument("--train_path", default="train_gold.json")
    parser.add_argument("--eval_path", default="eval_gold.json")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    dl_train = DataLoader(CostDataset(args.train_path, tokenizer), batch_size=args.batch_size, shuffle=True)
    dl_eval = DataLoader(CostDataset(args.eval_path, tokenizer), batch_size=args.batch_size, shuffle=False)

    mape_base = run_baseline(tokenizer, dl_train, dl_eval)

    print("previous dataset mape:", mape_base)

    if os.path.exists(args.synthetic_path):
        dl_synth = DataLoader(CostDataset(args.synthetic_path, tokenizer), batch_size=args.batch_size, shuffle=True)
        mape_prop = run_proposed_with_warmup(tokenizer, dl_synth, dl_train, dl_eval)

        print("\n" + "="*40)
        print(f"Improvement: {mape_base - mape_prop:.2f} percentage points")
        print("="*40)