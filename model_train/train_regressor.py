import argparse
import json
import math
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, DebertaV2Model

# --- 配置 ---
MODEL_NAME = "microsoft/mdeberta-v3-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_LLMS = 3


# ==========================================
# 1. 模型架构 (Cost Only)
# ==========================================
class CostRouter(nn.Module):
    def __init__(self, backbone_name, num_llms, hidden_dim=768):
        super().__init__()
        self.backbone = DebertaV2Model.from_pretrained(backbone_name)
        # 如需进一步省显存，可以打开 gradient checkpointing：
        # self.backbone.gradient_checkpointing_enable()

        self.llm_embedding = nn.Embedding(num_llms, hidden_dim)

        self.cost_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, input_ids, attention_mask, llm_ids):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # [CLS] token 表示整句
        query_emb = outputs.last_hidden_state[:, 0, :]
        llm_emb = self.llm_embedding(llm_ids)
        combined = torch.cat([query_emb, llm_emb], dim=1)
        return self.cost_head(combined).squeeze(-1)


# ==========================================
# 2. 数据加载器 (Cost Log-Transform)
# ==========================================
class CostDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=256):  # 用 256 减少显存
        with open(data_path, "r") as f:
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

        # Log Space Transform
        raw_cost = item.get("cost", 1e-10)
        if raw_cost <= 0:
            raw_cost = 1e-10
        scaled_cost = math.log(raw_cost)

        return (
            encoding.input_ids.squeeze(0),       # [seq_len]
            encoding.attention_mask.squeeze(0),  # [seq_len]
            torch.tensor(item["llm_id"], dtype=torch.long),
            torch.tensor(scaled_cost, dtype=torch.float),
        )


# ==========================================
# 3. 评估与训练逻辑
# ==========================================
def evaluate(model, dataloader):
    model.eval()
    preds_log, labels_log = [], []

    with torch.no_grad():
        for input_ids, mask, llm_ids, labels in dataloader:
            input_ids = input_ids.to(DEVICE)
            mask = mask.to(DEVICE)
            llm_ids = llm_ids.to(DEVICE)
            labels = labels.to(DEVICE)

            preds = model(input_ids, mask, llm_ids)
            preds_log.extend(preds.cpu().numpy())
            labels_log.extend(labels.cpu().numpy())

    # 计算 MAPE (Mean Absolute Percentage Error) in *real* cost space
    real_preds = np.exp(preds_log)
    real_labels = np.exp(labels_log)
    epsilon = 1e-10
    mape = np.mean(np.abs((real_labels - real_preds) / (real_labels + epsilon))) * 100.0
    return mape


def train_loop(model, train_loader, val_loader, optimizer, epochs, task_name):
    criterion = nn.MSELoss()
    best_mape = float("inf")
    best_model = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for input_ids, mask, llm_ids, labels in train_loader:
            input_ids = input_ids.to(DEVICE)
            mask = mask.to(DEVICE)
            llm_ids = llm_ids.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            preds = model(input_ids, mask, llm_ids)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_mape = evaluate(model, val_loader)
        print(
            f"[{task_name}] Ep {epoch}: "
            f"MSE Loss {total_loss/len(train_loader):.4f} | Val MAPE {val_mape:.2f}%"
        )

        if val_mape < best_mape:
            best_mape = val_mape
            best_model = copy.deepcopy(model.state_dict())

    if best_model is not None:
        model.load_state_dict(best_model)

    return best_mape


# ==========================================
# 4. 主程序（Baseline / Proposed 分开跑）
# ==========================================
if __name__ == "__main__":
    print(">>> RUNNING COST ROUTER SCRIPT (MSE + MAPE) <<<")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["baseline", "proposed"],
        default="baseline",
        help="baseline: 只在 gold 上训练; proposed: 先 synth 预训练再 gold finetune",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 显存友好的 batch_size
    bs_synth = 4
    bs_gold = 4

    dl_synth = DataLoader(
        CostDataset("synthetic_data.json", tokenizer),
        batch_size=bs_synth,
        shuffle=True,
        num_workers=0,
    )
    dl_train = DataLoader(
        CostDataset("train_gold.json", tokenizer),
        batch_size=bs_gold,
        shuffle=True,
        num_workers=0,
    )
    dl_eval = DataLoader(
        CostDataset("eval_gold.json", tokenizer),
        batch_size=bs_gold,
        shuffle=False,
        num_workers=0,
    )

    if args.mode == "baseline":
        print("\n=== Training Cost Router: Baseline (No Pretrain) ===")
        model_base = CostRouter(MODEL_NAME, NUM_LLMS).to(DEVICE)
        opt_base = torch.optim.AdamW(model_base.parameters(), lr=1e-4)
        mape_base = train_loop(model_base, dl_train, dl_eval, opt_base, epochs=20, task_name="Base")

        print(f"\nFinal Baseline MAPE: {mape_base:.2f}%")

    elif args.mode == "proposed":
        print("\n=== Training Cost Router: Proposed (Synth Pretrain + Finetune) ===")

        model_prop = CostRouter(MODEL_NAME, NUM_LLMS).to(DEVICE)

        # Phase 1: synth 预训练（只回归 cost）
        opt_pre = torch.optim.AdamW(model_prop.parameters(), lr=5e-5)
        train_loop(model_prop, dl_synth, dl_eval, opt_pre, epochs=10, task_name="Phase1")

        # Phase 2: gold finetune，backbone 用较小 LR，head/embedding 用较大 LR
        params = [
            {"params": model_prop.backbone.parameters(), "lr": 1e-5},
            {"params": model_prop.cost_head.parameters(), "lr": 1e-3},
            {"params": model_prop.llm_embedding.parameters(), "lr": 1e-3},
        ]
        opt_ft = torch.optim.AdamW(params)
        mape_prop = train_loop(model_prop, dl_train, dl_eval, opt_ft, epochs=20, task_name="Phase2")

        print(f"\nFinal Proposed MAPE: {mape_prop:.2f}%")
