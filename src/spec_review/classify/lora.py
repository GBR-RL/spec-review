"""LoRA fine-tuning of a small open LLM as a requirement classifier.

Qwen3-0.6B gets a classification head over the 12 classes; only low-rank adapters on the
attention projections and the head are trained (about 1% of the weights). The loss is weighted
by inverse class frequency, because portability or legal have a dozen examples each. One fold at
a time, so the five folds can train on five CPU runners in parallel.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from spec_review.classify import cv
from spec_review.config import RUNS, SEED

MODEL = "Qwen/Qwen3-0.6B"
LABELS = ("F", "A", "FT", "L", "LF", "MN", "O", "PE", "PO", "SC", "SE", "US")
OUT = RUNS / "lora"


def train_fold(
    fold: int,
    *,
    model_name: str = MODEL,
    epochs: int = 3,
    lr: float = 3e-4,
    rank: int = 16,
    batch_size: int = 8,
    max_length: int = 96,
    out: Path = OUT,
) -> Path:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    data, y, fold_of = cv.task_data("all-12")
    train, test = np.where(fold_of != fold)[0], np.where(fold_of == fold)[0]
    label_id = {lab: i for i, lab in enumerate(LABELS)}
    y_ids = np.array([label_id[v] for v in y])

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id=label_id,
        torch_dtype=torch.float32,
    )
    model.config.pad_token_id = tok.pad_token_id
    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=rank,
        lora_alpha=2 * rank,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    counts = np.bincount(y_ids[train], minlength=len(LABELS)).astype(float)
    weights = torch.tensor(
        counts.sum() / (len(LABELS) * np.maximum(counts, 1)), dtype=torch.float32
    )
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    steps = math.ceil(len(train) / batch_size) * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    texts = data.text.tolist()

    def batch(idx: np.ndarray) -> Any:
        return tok([texts[i] for i in idx], padding=True, truncation=True,
                   max_length=max_length, return_tensors="pt")  # fmt: skip

    t0 = time.perf_counter()
    model.train()
    for epoch in range(epochs):
        order = rng.permutation(train)
        total_loss = 0.0
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            enc = batch(idx)
            logits = model(**enc).logits
            loss = loss_fn(logits, torch.tensor(y_ids[idx]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total_loss += float(loss.detach()) * len(idx)
        print(f"fold {fold} epoch {epoch} loss {total_loss / len(train):.3f} "
              f"({time.perf_counter() - t0:.0f}s)")  # fmt: skip

    model.eval()
    preds = []
    with torch.inference_mode():
        for start in range(0, len(test), 32):
            idx = test[start : start + 32]
            preds.extend(model(**batch(idx)).logits.argmax(-1).tolist())
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"fold{fold}.parquet"
    pd.DataFrame(
        {
            "id": data.id.iloc[test].to_numpy(),
            "pred": [LABELS[p] for p in preds],
            "fold": fold,
            "train_seconds": round(time.perf_counter() - t0),
            "trainable_params": trainable,
            "total_params": total,
        }
    ).to_parquet(path, index=False)
    return path


def merge(out: Path = OUT) -> pd.DataFrame:
    parts = sorted(out.glob("fold*.parquet"))
    if len(parts) != cv.N_FOLDS:
        raise FileNotFoundError(f"expected {cv.N_FOLDS} fold files in {out}, found {len(parts)}")
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
