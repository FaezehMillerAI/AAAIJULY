import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
from pathlib import Path
from typing import Callable, Optional
from tqdm import tqdm

def train_model(
    model,
    train_dataset,
    val_dataset,
    epochs: int = 2,
    batch_size: int = 8,
    lr: float = 5e-5,
    fp16: bool = True,
    device: str = "cuda",
    checkpoint_dir: Optional[Path] = None,
    log_fn: Optional[Callable[[str], None]] = print,
):
    """
    Trains the VisionT5 model.
    """
    if log_fn is None:
        log_fn = lambda x: None

    if device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device(device if torch.cuda.is_available() else "cpu")
    log_fn(f"Using device: {device}")
    model.to(device)

    # ── Mixed precision setup (bfloat16 for T5 stability) ─────────────────
    autocast_dtype = torch.float32
    if fp16 and device.type == "cuda":
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            autocast_dtype = torch.bfloat16
            log_fn("Using bfloat16 mixed precision (stable for T5).")
        else:
            log_fn("Warning: bfloat16 not supported — falling back to float32.")
    else:
        log_fn("Using float32 precision.")

    autocast_kwargs = {"enabled": autocast_dtype != torch.float32}
    if autocast_dtype != torch.float32:
        autocast_kwargs["dtype"] = autocast_dtype

    # Optimize data loading (parallelize via workers and pin memory)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=2,
        pin_memory=True,
    )
    val_loader   = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs * max(1, len(train_loader)))

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        start_time = time.time()

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for batch in train_pbar:
            images               = batch["images"].to(device)
            encoder_input_ids    = batch["encoder_input_ids"].to(device)
            encoder_attention_mask = batch["encoder_attention_mask"].to(device)
            labels               = batch["labels"].to(device)

            optimizer.zero_grad()

            with torch.amp.autocast(device.type, **autocast_kwargs):
                outputs = model(
                    images=images,
                    encoder_input_ids=encoder_input_ids,
                    encoder_attention_mask=encoder_attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation step
        model.eval()
        total_val_loss = 0.0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]")
        with torch.no_grad():
            for batch in val_pbar:
                images               = batch["images"].to(device)
                encoder_input_ids    = batch["encoder_input_ids"].to(device)
                encoder_attention_mask = batch["encoder_attention_mask"].to(device)
                labels               = batch["labels"].to(device)

                with torch.amp.autocast(device.type, **autocast_kwargs):
                    outputs = model(
                        images=images,
                        encoder_input_ids=encoder_input_ids,
                        encoder_attention_mask=encoder_attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss
                total_val_loss += loss.item()
                val_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_val_loss = total_val_loss / len(val_loader)
        epoch_time = time.time() - start_time

        log_fn(
            f"Epoch {epoch}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Time: {epoch_time:.1f}s"
        )

        # Save best checkpoint
        if avg_val_loss < best_val_loss and checkpoint_dir is not None:
            best_val_loss = avg_val_loss
            log_fn(
                f"New best validation loss: {best_val_loss:.4f}. Saving checkpoint to {checkpoint_dir}"
            )
            model.save_checkpoint(str(checkpoint_dir))

    log_fn("Training completed.")
    return best_val_loss
