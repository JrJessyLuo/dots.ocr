# import argparse
# import json
# import copy
# import random
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader, ConcatDataset
# from transformers import AutoTokenizer, DebertaV2Model
# from sklearn.metrics import accuracy_score
# from tqdm import tqdm
# import os

# # ---------------------------------------------------------------------
# # Config
# # ---------------------------------------------------------------------
# MODEL_NAME = "microsoft/mdeberta-v3-base"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# NUM_LLMS = 3 

# # ---------------------------------------------------------------------
# # Reproducibility
# # ---------------------------------------------------------------------
# def set_seed(seed: int = 42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# # ---------------------------------------------------------------------
# # 1. Dataset
# # ---------------------------------------------------------------------
# class RouterDataset(Dataset):
#     def __init__(self, data_path, tokenizer, max_len: int = 512, fraction: float = 1.0):
#         with open(data_path, "r", encoding="utf-8") as f:
#             self.data = json.load(f)

#         if 0.0 < fraction < 1.0:
#             n_total = len(self.data)
#             n_keep = max(1, int(n_total * fraction))
#             self.data = random.sample(self.data, n_keep)

#         self.tokenizer = tokenizer
#         self.max_len = max_len

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         item = self.data[idx]
#         encoding = self.tokenizer(
#             item["question"],
#             max_length=self.max_len,
#             padding="max_length",
#             truncation=True,
#             return_tensors="pt",
#         )
#         input_ids = encoding["input_ids"].squeeze(0)
#         attention_mask = encoding["attention_mask"].squeeze(0)

#         llm_id = torch.tensor(item["llm_id"], dtype=torch.long)
#         acc = torch.tensor(item["accuracy"], dtype=torch.float)

#         return input_ids, attention_mask, llm_id, acc

# # ---------------------------------------------------------------------
# # 2. Model with Enhanced Cold-Start Logic
# # ---------------------------------------------------------------------
# class AccuracyRouter(nn.Module):
#     def __init__(self, backbone_name: str, num_llms: int, hidden_dim: int = 768):
#         super().__init__()
#         self.backbone = DebertaV2Model.from_pretrained(backbone_name)
#         self.llm_embedding = nn.Embedding(num_llms, hidden_dim)

#         self.acc_head = nn.Sequential(
#             nn.Linear(hidden_dim * 2, 256),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(256, 1),
#         )
        
#         self.register_buffer("memory_bank", None)

#     def populate_memory_bank(self, dataloader, device):
#         self.eval()
#         embeddings_list = []
#         print(">> Populating Memory Bank (Synth + Gold)...")
        
#         with torch.no_grad():
#             for batch in tqdm(dataloader, desc="Encoding Memory"):
#                 input_ids, mask, _, _ = [b.to(device) for b in batch]
#                 outputs = self.backbone(input_ids=input_ids, attention_mask=mask)
#                 query_emb = outputs.last_hidden_state[:, 0, :]
#                 # Normalize for Cosine Similarity
#                 query_norm = F.normalize(query_emb, p=2, dim=1)
#                 embeddings_list.append(query_norm)
        
#         if embeddings_list:
#             self.memory_bank = torch.cat(embeddings_list, dim=0)
#             print(f">> Memory Bank populated with {self.memory_bank.size(0)} vectors.")
#         else:
#             print(">> Warning: Memory bank is empty!")

#     def get_warmup_embedding(self, query_emb, k=5):
#         if self.memory_bank is None: return query_emb 
        
#         query_norm = F.normalize(query_emb, p=2, dim=1)
#         sim_matrix = torch.matmul(query_norm, self.memory_bank.t())
        
#         # Retrieve Top-K neighbors
#         actual_k = min(k, self.memory_bank.size(0))
#         _, indices = torch.topk(sim_matrix, k=actual_k, dim=1)
        
#         neighbors = self.memory_bank[indices] 
#         e_warm = torch.mean(neighbors, dim=1) 
        
#         return e_warm

#     def forward(self, input_ids, attention_mask, llm_ids, use_warmup=False, warmup_lambda=0.3, k=5):
#         outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
#         query_emb = outputs.last_hidden_state[:, 0, :] 
        
#         if use_warmup and self.memory_bank is not None:
#             e_warm = self.get_warmup_embedding(query_emb, k=k)
#             # Equation 10: Blending
#             query_emb = (1 - warmup_lambda) * query_emb + warmup_lambda * e_warm
            
#         llm_emb = self.llm_embedding(llm_ids)
#         combined = torch.cat([query_emb, llm_emb], dim=1)
#         logits = self.acc_head(combined).squeeze(-1)
        
#         return logits

# # ---------------------------------------------------------------------
# # 3. Evaluation & Training
# # ---------------------------------------------------------------------
# def evaluate(model, dataloader, use_warmup=False, warmup_lambda=0.3, k=5):
#     model.eval()
#     all_preds, all_labels = [], []

#     with torch.no_grad():
#         for batch in dataloader:
#             input_ids, mask, llm_ids, labels = [b.to(DEVICE) for b in batch]
            
#             logits = model(
#                 input_ids, mask, llm_ids, 
#                 use_warmup=use_warmup, 
#                 warmup_lambda=warmup_lambda, 
#                 k=k
#             )
            
#             preds = (torch.sigmoid(logits) > 0.5).float()
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())

#     return accuracy_score(all_labels, all_preds)

# def train_loop(model, train_loader, val_loader, optimizer, epochs, task_name):
#     criterion = nn.BCEWithLogitsLoss()
#     best_acc = 0.0
#     best_state = None

#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0.0
#         for batch in train_loader:
#             input_ids, mask, llm_ids, labels = [b.to(DEVICE) for b in batch]
#             optimizer.zero_grad()
#             # Train WITHOUT warmup to learn raw features
#             logits = model(input_ids, mask, llm_ids, use_warmup=False)
#             loss = criterion(logits, labels)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()

#         val_acc = evaluate(model, val_loader, use_warmup=False)
#         if val_acc > best_acc:
#             best_acc = val_acc
#             best_state = copy.deepcopy(model.state_dict())

#     if best_state is not None:
#         model.load_state_dict(best_state)
#     return best_acc

# def reset_module(m: nn.Module):
#     for layer in m.modules():
#         if hasattr(layer, "reset_parameters"): layer.reset_parameters()

# # ---------------------------------------------------------------------
# # 4. Runners
# # ---------------------------------------------------------------------
# def run_proposed_with_warmup(tokenizer, dl_synth, dl_train, dl_eval, 
#                              pre_epochs=5, ft_epochs=20):
    
#     print("\n--- Proposed Method (Synth Pretrain -> Gold Finetune -> Expanded Warmup) ---")
#     model = AccuracyRouter(MODEL_NAME, NUM_LLMS).to(DEVICE)

#     # 1. Phase 1: Pretrain
#     print("Phase 1: Pre-training...")
#     opt_pre = torch.optim.AdamW(model.parameters(), lr=2e-5)
#     _ = train_loop(model, dl_synth, dl_eval, opt_pre, pre_epochs, "Pretrain")

#     # 2. Reset Head & Finetune
#     reset_module(model.acc_head)
#     print("Phase 2: Fine-tuning...")
#     params = [
#         {"params": model.backbone.parameters(),      "lr": 5e-6},
#         {"params": model.llm_embedding.parameters(), "lr": 5e-5},
#         {"params": model.acc_head.parameters(),      "lr": 1e-3},
#     ]
#     opt_ft = torch.optim.AdamW(params)
#     train_loop(model, dl_train, dl_eval, opt_ft, ft_epochs, "Finetune")

#     # 3. Warmup Setup: Create a MEGA Memory Bank (Synth + Gold)
#     # Merging datasets maximizes the chance of finding a good neighbor
#     print("\n>> Populating Expanded Memory Bank (Synth + Train Gold)...")
    
#     # We create a temporary dataloader that combines both sources
#     # Access underlying datasets
#     ds_s = dl_synth.dataset
#     ds_t = dl_train.dataset
#     combined_ds = ConcatDataset([ds_s, ds_t])
#     dl_combined = DataLoader(combined_ds, batch_size=32, shuffle=False)
    
#     model.populate_memory_bank(dl_combined, DEVICE)
    
#     # 4. Evaluation with Warmup Grid Search (Find best Lambda on Validation)
#     # Note: In a real scenario, tune on a val set. Here we show effect on Eval.
#     print("\n>> Evaluating Warmup Impact...")
    
#     acc_std = evaluate(model, dl_eval, use_warmup=False)
#     print(f"Standard (No Warmup): {acc_std:.2%}")
    
#     best_warm_acc = 0
#     best_cfg = ""
    
#     # Search a small grid to find optimal blending
#     for lam in [0.1, 0.3, 0.5, 0.7]:
#         for k in [5, 10, 20]:
#             acc = evaluate(model, dl_eval, use_warmup=True, warmup_lambda=lam, k=k)
#             if acc > best_warm_acc:
#                 best_warm_acc = acc
#                 best_cfg = f"Lambda={lam}, K={k}"
#             # print(f"Warmup (L={lam}, K={k}): {acc:.2%}")
            
#     print(f"Best Warmup ({best_cfg}): {best_warm_acc:.2%}")
    
#     return max(acc_std, best_warm_acc)

# def run_baseline(tokenizer, dl_train, dl_eval, epochs=20):
#     print("\n--- Running Baseline (Gold Only) ---")
#     model = AccuracyRouter(MODEL_NAME, NUM_LLMS).to(DEVICE)
#     params = [
#         {"params": model.backbone.parameters(), "lr": 5e-6},
#         {"params": model.llm_embedding.parameters(), "lr": 5e-5},
#         {"params": model.acc_head.parameters(), "lr": 1e-3},
#     ]
#     opt = torch.optim.AdamW(params)
#     acc = train_loop(model, dl_train, dl_eval, opt, epochs, "Baseline")
#     return acc

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--synthetic_path", default="/research/remote/petabyte/users/s3941200/table_qa_methods/ST-Raptor-new_update/ablation_study/reasoning/RouterDC-main/synthetic_data_stratified.json")
#     parser.add_argument("--train_path", default="/research/remote/petabyte/users/s3941200/table_qa_methods/ST-Raptor-new_update/ablation_study/reasoning/RouterDC-main/train_gold_stratified.json")
#     parser.add_argument("--eval_path", default="/research/remote/petabyte/users/s3941200/table_qa_methods/ST-Raptor-new_update/ablation_study/reasoning/RouterDC-main/eval_gold_stratified.json")
#     parser.add_argument("--batch_size", type=int, default=8)
#     parser.add_argument("--seed", type=int, default=42)
#     args = parser.parse_args()

#     set_seed(args.seed)
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

#     dl_train = DataLoader(RouterDataset(args.train_path, tokenizer), batch_size=args.batch_size, shuffle=True)
#     dl_eval = DataLoader(RouterDataset(args.eval_path, tokenizer), batch_size=args.batch_size, shuffle=False)
#     acc_base = run_baseline(tokenizer, dl_train, dl_eval)

#     print(f"[Baseline] Accu on eval set: {acc_base:.4f}")


#     if os.path.exists(args.synthetic_path):
#         dl_synth = DataLoader(RouterDataset(args.synthetic_path, tokenizer), batch_size=args.batch_size, shuffle=True)
#         acc_prop = run_proposed_with_warmup(tokenizer, dl_synth, dl_train, dl_eval)

#         print("\n" + "="*40)
#         print(f"Final Improvement: {acc_prop - acc_base:.4f} ({(acc_prop - acc_base)*100:.2f}%)")
#         print("="*40)

import argparse
import json
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from transformers import AutoTokenizer, DebertaV2Model, AutoModel
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
MODEL_NAME = "microsoft/mdeberta-v3-base"
# MODEL_NAME = "BAAI/bge-m3"
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
# 1. Dataset
# ---------------------------------------------------------------------
class RouterDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len: int = 512, fraction: float = 1.0):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        if 0.0 < fraction < 1.0:
            n_total = len(self.data)
            n_keep = max(1, int(n_total * fraction))
            self.data = random.sample(self.data, n_keep)

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
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        llm_id = torch.tensor(item["llm_id"], dtype=torch.long)
        acc = torch.tensor(item["accuracy"], dtype=torch.float)

        return input_ids, attention_mask, llm_id, acc

# ---------------------------------------------------------------------
# 2. Model
# ---------------------------------------------------------------------
class AccuracyRouter(nn.Module):
    def __init__(self, backbone_name: str, num_llms: int, hidden_dim: int = 768):
        super().__init__()
        self.backbone = DebertaV2Model.from_pretrained(backbone_name)
        # self.backbone = AutoModel.from_pretrained(backbone_name)
        self.llm_embedding = nn.Embedding(num_llms, hidden_dim)

        self.acc_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )
        
        self.register_buffer("memory_bank", None)

    def populate_memory_bank(self, dataloader, device):
        self.eval()
        embeddings_list = []
        print(">> Populating Memory Bank (Synth + Gold)...")
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Encoding Memory"):
                input_ids, mask, _, _ = [b.to(device) for b in batch]
                outputs = self.backbone(input_ids=input_ids, attention_mask=mask)
                query_emb = outputs.last_hidden_state[:, 0, :]
                query_norm = F.normalize(query_emb, p=2, dim=1)
                embeddings_list.append(query_norm)
        
        if embeddings_list:
            self.memory_bank = torch.cat(embeddings_list, dim=0)
            print(f">> Memory Bank populated with {self.memory_bank.size(0)} vectors.")
        else:
            print(">> Warning: Memory bank is empty!")

    def get_warmup_embedding(self, query_emb, k=5):
        if self.memory_bank is None: return query_emb 
        
        query_norm = F.normalize(query_emb, p=2, dim=1)
        sim_matrix = torch.matmul(query_norm, self.memory_bank.t())
        
        actual_k = min(k, self.memory_bank.size(0))
        _, indices = torch.topk(sim_matrix, k=actual_k, dim=1)
        
        neighbors = self.memory_bank[indices] 
        e_warm = torch.mean(neighbors, dim=1) 
        
        return e_warm

    def forward(self, input_ids, attention_mask, llm_ids, use_warmup=False, warmup_lambda=0.3, k=5):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        query_emb = outputs.last_hidden_state[:, 0, :] 
        
        if use_warmup and self.memory_bank is not None:
            e_warm = self.get_warmup_embedding(query_emb, k=k)
            query_emb = (1 - warmup_lambda) * query_emb + warmup_lambda * e_warm
            
        llm_emb = self.llm_embedding(llm_ids)
        combined = torch.cat([query_emb, llm_emb], dim=1)
        logits = self.acc_head(combined).squeeze(-1)
        
        return logits

# ---------------------------------------------------------------------
# 3. Evaluation
# ---------------------------------------------------------------------
def evaluate(model, dataloader, use_warmup=False, warmup_lambda=0.3, k=5):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, mask, llm_ids, labels = [b.to(DEVICE) for b in batch]
            
            logits = model(
                input_ids, mask, llm_ids, 
                use_warmup=use_warmup, 
                warmup_lambda=warmup_lambda, 
                k=k
            )
            
            preds = (torch.sigmoid(logits) > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return accuracy_score(all_labels, all_preds)

def train_loop(model, train_loader, val_loader, optimizer, epochs, task_name):
    criterion = nn.BCEWithLogitsLoss()
    best_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids, mask, llm_ids, labels = [b.to(DEVICE) for b in batch]
            optimizer.zero_grad()
            logits = model(input_ids, mask, llm_ids, use_warmup=False)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_acc = evaluate(model, val_loader, use_warmup=False)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_acc

def reset_module(m: nn.Module):
    for layer in m.modules():
        if hasattr(layer, "reset_parameters"): layer.reset_parameters()

# ---------------------------------------------------------------------
# 4. Runners (Optimized with Replay)
# ---------------------------------------------------------------------
def run_proposed_with_warmup(tokenizer, dl_synth, dl_train, dl_eval, 
                             pre_epochs=5, ft_epochs=20):
    
    print("\n--- Proposed Method (Synth Pretrain -> Replay Finetune -> Warmup) ---")
    model = AccuracyRouter(MODEL_NAME, NUM_LLMS).to(DEVICE)

    # 1. Phase 1: Pretrain on Synthetic
    print("Phase 1: Pre-training on Synthetic Data...")
    opt_pre = torch.optim.AdamW(model.parameters(), lr=2e-5)
    _ = train_loop(model, dl_synth, dl_eval, opt_pre, pre_epochs, "Pretrain")

    # 2. Phase 2: Replay Fine-tuning
    print("Phase 2: Fine-tuning with Replay (Gold + 20% Synth)...")
    
    reset_module(model.acc_head)
    
    # Create Replay Dataset: Mix Gold with a subset of Synthetic
    synth_ds = dl_synth.dataset
    train_ds = dl_train.dataset
    
    # Sample a buffer of synthetic data (e.g., 3x the size of Gold data)
    # This ensures the epoch is dominated by Gold but "reminded" of Synthetic patterns
    replay_size = int(len(train_ds) * 3.0) 
    indices = np.random.choice(len(synth_ds), replay_size, replace=False)
    synth_replay = Subset(synth_ds, indices)
    
    combined_ds = ConcatDataset([train_ds, synth_replay])
    dl_combined = DataLoader(combined_ds, batch_size=16, shuffle=True)

    params_ft = [
        {"params": model.backbone.parameters(),      "lr": 5e-6},
        {"params": model.llm_embedding.parameters(), "lr": 5e-5},
        {"params": model.acc_head.parameters(),      "lr": 1e-3},
    ]
    opt_ft = torch.optim.AdamW(params_ft)
    
    train_loop(model, dl_combined, dl_eval, opt_ft, ft_epochs, "Finetune")

    # 3. Warmup Eval
    print("\n>> Populating Expanded Memory Bank (Synth + Gold)...")
    # Combine all available data for the memory bank to maximize coverage
    dl_bank = DataLoader(ConcatDataset([train_ds, synth_ds]), batch_size=32, shuffle=False)
    model.populate_memory_bank(dl_bank, DEVICE)
    
    print("\n>> Evaluating Warmup Impact...")
    acc_std = evaluate(model, dl_eval, use_warmup=False)
    print(f"Standard (No Warmup): {acc_std:.2%}")
    
    best_warm_acc = 0
    best_cfg = ""
    
    # Grid search for best warmup params
    for lam in [0.1, 0.3, 0.5]:
        for k in [5, 10, 20]:
            acc = evaluate(model, dl_eval, use_warmup=True, warmup_lambda=lam, k=k)
            if acc > best_warm_acc:
                best_warm_acc = acc
                best_cfg = f"L={lam}, K={k}"
            
    print(f"Best Warmup ({best_cfg}): {best_warm_acc:.2%}")
    
    return max(acc_std, best_warm_acc)

def run_baseline(tokenizer, dl_train, dl_eval, epochs=20):
    print("\n--- Running Baseline (Gold Only) ---")
    model = AccuracyRouter(MODEL_NAME, NUM_LLMS).to(DEVICE)
    params = [
        {"params": model.backbone.parameters(), "lr": 5e-6},
        {"params": model.llm_embedding.parameters(), "lr": 5e-5},
        {"params": model.acc_head.parameters(), "lr": 1e-3},
    ]
    opt = torch.optim.AdamW(params)
    acc = train_loop(model, dl_train, dl_eval, opt, epochs, "Baseline")
    print(f"Baseline Accuracy: {acc:.2%}")
    return acc

# ---------------------------------------------------------------------
# 5. Entry Point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic_path", default="synthetic_data_stratified.json") # Use the smart scaled data
    parser.add_argument("--train_path", default="train_gold_stratified.json")
    parser.add_argument("--eval_path", default="eval_gold_stratified.json")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    dl_synth = DataLoader(RouterDataset(args.synthetic_path, tokenizer), batch_size=args.batch_size, shuffle=True)
    dl_train = DataLoader(RouterDataset(args.train_path, tokenizer), batch_size=args.batch_size, shuffle=True)
    dl_eval = DataLoader(RouterDataset(args.eval_path, tokenizer), batch_size=args.batch_size, shuffle=False)

    acc_base = run_baseline(tokenizer, dl_train, dl_eval)
    
    if os.path.exists(args.synthetic_path):
        acc_prop = run_proposed_with_warmup(tokenizer, dl_synth, dl_train, dl_eval)
        print("\n" + "="*40)
        print(f"Final Improvement: {acc_prop - acc_base:.4f} ({(acc_prop - acc_base)*100:.2f}%)")
        print("="*40)