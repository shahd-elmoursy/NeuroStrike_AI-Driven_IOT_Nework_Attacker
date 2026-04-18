import gc
import json
import pickle
import shutil
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from pathlib import Path

warnings.filterwarnings("ignore")

try:
    from ctgan import CTGAN
except ImportError:
    raise ImportError("\nRun: pip install ctgan sdv\n")

import torch

# PATHS — COLAB
BASE_DIR     = Path('/content/CTGAN_Ready_NeuroStrike_v2')
OUTPUT_DIR   = Path('/content/CTGAN_Models_NeuroStrike_v2')
DRIVE_OUTPUT = Path('/content/drive/MyDrive/NeuroStrike/CTGAN_Models_NeuroStrike_v2')

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DRIVE_OUTPUT.mkdir(parents=True, exist_ok=True)


RESUME_FROM_BLOCK = 0    # global default — 0 means start fresh for all attacks
ATTACK_RESUME     = {}   # per-attack override — takes priority over RESUME_FROM_BLOCK

# FEATURE DEFINITIONS
ATTACK_TYPES = [
    "Basic_Connect_Flooding",
    "Connect_Flooding_with_WILL_payload",
    "Delayed_Connect_Flooding",
    "Invalid_Subscription_Flooding",
    "SYN_TCP_Flooding",
]

CONTINUOUS_FEATURES = ["delta_time", "packet_len", "payload_len", "tcp_window_size"]
BINARY_FEATURES     = ["flag_syn", "flag_ack", "flag_fin", "flag_rst",
                        "flag_psh", "flag_urg", "port_direction"]
ALL_FEATURES        = CONTINUOUS_FEATURES + BINARY_FEATURES
DISCRETE_COLUMNS    = BINARY_FEATURES + ["attack_label"]

# TRAINING CONFIG

EPOCHS_PER_BLOCK = 50     # epochs per fit() call
N_BLOCKS         = 3      # total blocks → 150 honest epochs
TOTAL_EPOCHS     = EPOCHS_PER_BLOCK * N_BLOCKS

CTGAN_CONFIG = {
    "epochs":              EPOCHS_PER_BLOCK,  # per block — NOT total
    "batch_size":          500,
    "generator_dim":       (256, 256, 256),
    "discriminator_dim":   (256, 256, 256),
    "generator_lr":        2e-4,
    "discriminator_lr":    2e-4,
    "discriminator_steps": 1,
    "log_frequency":       True,
    "pac":                 2,      
    "cuda":                True,
}

# Collapse thresholds
COLLAPSE_STD_MIN   = 0.005   # fine — log-transformed delta_time std ~1.0+
COLLAPSE_COMBO_MIN = 3

# HELPERS
def elapsed(start: float) -> str:
    s = time.time() - start
    return f"{s:.1f}s" if s < 60 else f"{int(s//60)}m {int(s%60)}s"


def mirror(filename: str):
    src = OUTPUT_DIR / filename
    dst = DRIVE_OUTPUT / filename
    if src.exists():
        shutil.copy2(src, dst)
        print(f"    Mirrored → Drive/{filename}")


def save_and_mirror(model, filename: str):
    path = OUTPUT_DIR / filename
    with open(path, "wb") as fp:
        pickle.dump(model, fp)
    mirror(filename)


def block_ckpt_name(safe_name: str, block: int) -> str:
    return f"ckpt_{safe_name}_block_{block:03d}.pkl"


def load_block_checkpoint(safe_name: str, block: int):
    fname = block_ckpt_name(safe_name, block)
    local = OUTPUT_DIR / fname
    on_drive = DRIVE_OUTPUT / fname

    if local.exists():
        print(f"  Loading local checkpoint: {fname}")
        with open(local, "rb") as fp:
            return pickle.load(fp)

    if on_drive.exists():
        print(f"  Loading Drive checkpoint: {fname} (copying to local...)")
        shutil.copy2(on_drive, local)
        with open(local, "rb") as fp:
            return pickle.load(fp)

    raise FileNotFoundError(
        f"\nCheckpoint not found for block {block}: {fname}"
        f"\nLooked in:\n  {local}\n  {on_drive}"
        f"\nCheck your ATTACK_RESUME setting — "
        f"block {block} may not have completed before disconnection."
    )


# COLLAPSE CHECK
def check_collapse(model, attack_name: str, block: int) -> dict:
    epoch_reached = block * EPOCHS_PER_BLOCK
    print(f"\n  Collapse check — block {block}/{N_BLOCKS} "
          f"({epoch_reached}/{TOTAL_EPOCHS} epochs):")

    try:
        sample = model.sample(1000)
    except Exception as e:
        print(f"    Sample generation failed: {e}")
        return {"status": "ERROR", "block": block, "signals": 3,
                "combos": 0, "low_var": [], "no_cov": []}

    signals = 0

    # Signal 1: variance
    # With log transform, delta_time std should be ~1.0+ not 0.005
    low_var = []
    for feat in CONTINUOUS_FEATURES:
        if feat in sample.columns:
            std = float(sample[feat].std())
            if std < COLLAPSE_STD_MIN:
                low_var.append(f"{feat}(std={std:.5f})")
    if low_var:
        signals += 1

    # Signal 2: binary coverage
    no_cov = []
    for feat in ["flag_syn", "flag_ack", "flag_fin", "flag_rst", "flag_psh"]:
        if feat in sample.columns:
            vals = sample[feat].dropna()
            if not ((vals == 0).any() and (vals == 1).any()):
                no_cov.append(feat)
    if no_cov:
        signals += 1

    # Signal 3: flag diversity
    flag_cols = [f for f in ["flag_syn","flag_ack","flag_fin","flag_rst","flag_psh"]
                 if f in sample.columns]
    combos = 0
    if flag_cols:
        combos = int(sample[flag_cols].astype(int)
                     .apply(tuple, axis=1).nunique())
        if combos < COLLAPSE_COMBO_MIN:
            signals += 1

    # Determine status
    if signals == 0:
        status = "OK"
    elif signals == 1:
        status = "WARNING"
    else:
        status = "CRITICAL"

    icon = {"OK": "✅", "WARNING": "⚠️ ", "CRITICAL": "🔴"}.get(status, "?")

    # Print feature stats
    print(f"    {icon} Status: {status}")
    print(f"    {'Feature':<22} {'mean':>10} {'std':>10}")
    print(f"    {'─'*44}")
    for feat in CONTINUOUS_FEATURES:
        if feat in sample.columns:
            print(f"    {feat:<22} {sample[feat].mean():>10.4f} "
                  f"{sample[feat].std():>10.4f}")
    print(f"    flag combinations : {combos} unique")

    if low_var:
        print(f"    low_var  : {low_var}")
    if no_cov:
        print(f"    no_cov   : {no_cov}")

    if status == "CRITICAL":
        print(f"\n    🔴 CRITICAL collapse — "
              f"use block {max(1, block-1)} checkpoint for generation")
        print(f"       Consider extending pac or checking input data")

    del sample
    gc.collect()

    return {
        "status":  status,
        "block":   block,
        "epochs":  epoch_reached,
        "signals": signals,
        "combos":  combos,
        "low_var": low_var,
        "no_cov":  no_cov,
    }


# TRAIN ONE ATTACK TYPE
def train_one(attack_name: str, attack_idx: int) -> dict:
    safe    = attack_name.replace(" ", "_")
    t_start = time.time()

    print(f"\n{'=' * 65}")
    print(f"  [{attack_idx+1}/{len(ATTACK_TYPES)}] {attack_name}")
    print(f"{'=' * 65}")

    # Load data
    csv_path = BASE_DIR / "per_attack" / f"{attack_name}_train.csv"
    if not csv_path.exists():
        print(f"  Not found: {csv_path} — skipping")
        return {"attack": attack_name, "status": "SKIPPED"}

    df = pd.read_csv(csv_path)
    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        print(f"  Missing columns: {missing}")
        return {"attack": attack_name, "status": "SKIPPED"}

    df = df[ALL_FEATURES + ["attack_label"]].dropna()
    print(f"  Rows: {len(df):,}  |  Features: {len(ALL_FEATURES)}")

    # Verify log transform
    dt_std = float(df["delta_time"].std())
    if dt_std < 0.5:
        print(f"\n  ⚠️  WARNING: delta_time std={dt_std:.4f}")
        print(f"       Expected ~1.0+ after log transform.")
        print(f"       Check BASE_DIR points at Code 2 v3 output: {BASE_DIR}")
    else:
        print(f"  ✓ delta_time std={dt_std:.4f} — log transform confirmed")

    # Feature preview
    print(f"\n  Feature preview (log-transformed values):")
    for feat in CONTINUOUS_FEATURES:
        print(f"    {feat:<20} mean={df[feat].mean():.4f}  "
              f"std={df[feat].std():.4f}  "
              f"min={df[feat].min():.4f}  "
              f"max={df[feat].max():.4f}")
    for feat in ["flag_syn", "flag_ack", "flag_rst"]:
        print(f"    {feat:<20} rate={df[feat].mean():.3f}")

    # Resume setting for this attack 
    resume_block = ATTACK_RESUME.get(attack_name, RESUME_FROM_BLOCK)

    print(f"\n  Training plan: {N_BLOCKS} blocks × {EPOCHS_PER_BLOCK} epochs "
          f"= {TOTAL_EPOCHS} total honest epochs")
    print(f"  pac={CTGAN_CONFIG['pac']} | "
          f"batch={CTGAN_CONFIG['batch_size']} | "
          f"cuda={CTGAN_CONFIG['cuda']}")
    if resume_block > 0:
        print(f"  Resuming from block {resume_block} "
              f"(epoch {resume_block * EPOCHS_PER_BLOCK})")
    print(f"  {'─' * 55}")

    # Block training loop
    model        = None
    block_checks = []
    t_blocks     = time.time()
    best_block   = 0
    best_status  = "NONE"

    for block in range(1, N_BLOCKS + 1):
        epoch_start = (block - 1) * EPOCHS_PER_BLOCK
        epoch_end   = block * EPOCHS_PER_BLOCK

        # Skip completed blocks on resume
        if block <= resume_block:
            print(f"\n  Block {block}/{N_BLOCKS} "
                  f"(epochs {epoch_start+1}–{epoch_end}) — SKIPPING (completed)")
            model = load_block_checkpoint(safe, block)
            block_checks.append({
                "status": "SKIPPED", "block": block,
                "epochs": epoch_end, "signals": 0,
                "combos": 0, "low_var": [], "no_cov": []
            })
            continue

        # Run this block
        print(f"\n  Block {block}/{N_BLOCKS} "
              f"(epochs {epoch_start+1}–{epoch_end})")
        print(f"  elapsed so far: {elapsed(t_start)}")

        if model is None:
            # First block: create the model
            print(f"  Creating new CTGAN (pac={CTGAN_CONFIG['pac']})...")
            model = CTGAN(**CTGAN_CONFIG)
        else:
            # Subsequent blocks: reuse the existing model object
            # fit() will CONTINUE from current weights — not reinitialize
            print(f"  Continuing from block {block-1} "
                  f"(epoch {epoch_start}) weights...")

        t_block = time.time()
        model.fit(df, discrete_columns=DISCRETE_COLUMNS)
        print(f"  Block {block} fit() done — {elapsed(t_block)}")

        # Collapse check 
        check = check_collapse(model, attack_name, block)
        block_checks.append(check)

        # Save block checkpoint to local + Drive
        ckpt_fname = block_ckpt_name(safe, block)
        save_and_mirror(model, ckpt_fname)
        print(f"   Block {block} checkpoint → Drive  (recovery point)")

        # Track best block (prefer OK over WARNING over CRITICAL)
        status_rank = {"OK": 0, "WARNING": 1, "CRITICAL": 2, "ERROR": 3}
        if (best_block == 0 or
                status_rank.get(check["status"], 3) <=
                status_rank.get(best_status, 3)):
            best_block  = block
            best_status = check["status"]

        # Stop on critical collapse
        if check["status"] == "CRITICAL":
            print(f"\n  🔴 Critical collapse at block {block} — stopping early")
            print(f"     Best block was {best_block} ({best_status})")
            print(f"     Load ckpt_{safe}_block_{best_block:03d}.pkl for generation")
            break

        # ETA
        blocks_done = block - resume_block
        blocks_left = N_BLOCKS - block
        if blocks_done > 0 and blocks_left > 0:
            time_per_block = (time.time() - t_blocks) / blocks_done
            eta_min = int(time_per_block * blocks_left / 60)
            print(f"  ETA for remaining {blocks_left} block(s): ~{eta_min}m")

    # Save final model
    final_fname = f"ctgan_model_{safe}.pkl"
    save_and_mirror(model, final_fname)

    # Final sanity check
    print(f"\n  Final sanity check (1000 samples, log-space):")
    flag_combos = 0
    try:
        sample = model.sample(1000)
        print(f"  {'Feature':<22} {'mean':>10} {'std':>10}")
        print(f"  {'─'*44}")
        for feat in CONTINUOUS_FEATURES:
            if feat in sample.columns:
                print(f"  {feat:<22} {sample[feat].mean():>10.4f} "
                      f"{sample[feat].std():>10.4f}")
        fc = [f for f in ["flag_syn","flag_ack","flag_fin","flag_rst","flag_psh"]
              if f in sample.columns]
        if fc:
            flag_combos = int(sample[fc].astype(int)
                              .apply(tuple, axis=1).nunique())
        print(f"  flag combinations : {flag_combos} unique")
        del sample
        gc.collect()
    except Exception as e:
        print(f"  ⚠️  Sanity check failed: {e}")

    # Determine overall status
    real_checks = [c for c in block_checks if c["status"] != "SKIPPED"]
    statuses    = [c["status"] for c in real_checks]
    if "CRITICAL" in statuses:
        final_status = "CRITICAL"
    elif "WARNING" in statuses:
        final_status = "WARNING"
    elif all(s == "OK" for s in statuses):
        final_status = "OK"
    else:
        final_status = "UNKNOWN"

    print(f"\n   Final model    : {final_fname}")
    print(f"   Block statuses : {statuses}")
    print(f"   Overall        : {final_status}")
    print(f"    Total Time         : {elapsed(t_start)}")

    if final_status == "WARNING":
        print(f"\n    To extend training by 50 more epochs:")
        print(f"     Set N_BLOCKS = 4")
        print(f"     Set ATTACK_RESUME = {{'{attack_name}': 3}}")
        print(f"     Re-run — block 4 will train on top of existing weights")

    # Save collapse log
    log = {
        "attack_name":      attack_name,
        "total_epochs":     TOTAL_EPOCHS,
        "epochs_per_block": EPOCHS_PER_BLOCK,
        "n_blocks":         N_BLOCKS,
        "block_checks":     block_checks,
        "final_status":     final_status,
        "best_block":       best_block,
        "resume_block":     resume_block,
        "flag_combos":      flag_combos,
    }
    log_fname = f"collapse_log_{safe}.json"
    with open(OUTPUT_DIR / log_fname, "w") as fp:
        json.dump(log, fp, indent=2)
    mirror(log_fname)

    del model, df
    gc.collect()

    return {
        "attack":          attack_name,
        "status":          final_status,
        "total_epochs":    TOTAL_EPOCHS,
        "block_checks":    block_checks,
        "flag_combos":     flag_combos,
        "training_time_s": time.time() - t_start,
    }


# MAIN
def main():
    total_start = time.time()

    gpu_name = (torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else "CPU — no GPU detected")

    print("=" * 65)
    print("NeuroStrike - Code 3: CTGAN Training")
    print("=" * 65)
    print(f"GPU    : {gpu_name}")
    print(f"Data   : {BASE_DIR}  (local SSD)")
    print(f"Output : {OUTPUT_DIR}  (local SSD)")
    print(f"Backup : {DRIVE_OUTPUT}  (Drive mirror)")
    print(f"\nTraining structure:")
    print(f"  {N_BLOCKS} blocks × {EPOCHS_PER_BLOCK} epochs = {TOTAL_EPOCHS} "
          f"total HONEST epochs")
    print(f"\nDisconnection recovery:")
    print(f"  Drive gets a checkpoint after every {EPOCHS_PER_BLOCK} epochs")
    print(f"  Worst case data loss: {EPOCHS_PER_BLOCK} epochs (~35 min on A100)")
    print(f"\nCTGAN config:")
    for k, v in CTGAN_CONFIG.items():
        note = ""
        if k == "pac":
            note = "  ← KEY FIX (was 10)"
        elif k == "epochs":
            note = f"  ← per block, {TOTAL_EPOCHS} total"
        print(f"  {k:<25} : {v}{note}")
    print(f"\nResume settings:")
    print(f"  RESUME_FROM_BLOCK = {RESUME_FROM_BLOCK}")
    print(f"  ATTACK_RESUME     = {ATTACK_RESUME if ATTACK_RESUME else '{} (fresh start)'}")
    print("=" * 65)

    # Verify Colab local data
    if not BASE_DIR.exists():
        raise RuntimeError(
            f"\nLocal data not found: {BASE_DIR}"
            f"\nCopy from Drive to local SSD first:"
            f"\n  !cp -r /content/drive/MyDrive/NeuroStrike/"
            f"CTGAN_Ready_NeuroStrike_v3 /content/"
        )

    print("\nVerifying input files...")
    missing_inputs = []
    for name in ATTACK_TYPES:
        p = BASE_DIR / "per_attack" / f"{name}_train.csv"
        if not p.exists():
            missing_inputs.append(name)
        else:
            print(f"  ✓ {name}_train.csv  ({p.stat().st_size/1e6:.1f} MB)")
    if missing_inputs:
        raise FileNotFoundError(
            f"\nMissing per-attack CSVs: {missing_inputs}"
            f"\nRun Code 2 v3 first."
        )

    # Train all attack types
    all_results = []
    for idx, attack_name in enumerate(ATTACK_TYPES):
        result = train_one(attack_name, idx)
        all_results.append(result)
        gc.collect()

    # Combined model — single honest fit(), no blocks needed
    print(f"\n{'=' * 65}")
    print(f"[Combined] Training on all attack types...")
    print(f"{'=' * 65}")
    combined_csv = BASE_DIR / "train_packets.csv"
    if combined_csv.exists():
        t0 = time.time()
        df_all = pd.read_csv(combined_csv,
                             usecols=ALL_FEATURES + ["attack_label"]).dropna()
        print(f"  Rows: {len(df_all):,}")
        combined_model = CTGAN(**CTGAN_CONFIG)
        combined_model.fit(df_all, discrete_columns=DISCRETE_COLUMNS)
        save_and_mirror(combined_model, "ctgan_model_combined.pkl")
        print(f"  ✓ Combined model saved — {elapsed(t0)}")
        del df_all, combined_model
        gc.collect()
    else:
        print(f"  ⚠️  train_packets.csv not found — skipping combined model")

    # Save metadata
    metadata = {
        "model_type":         "CTGAN — honest block training",
        "pac":                CTGAN_CONFIG["pac"],
        "total_epochs":       TOTAL_EPOCHS,
        "epochs_per_block":   EPOCHS_PER_BLOCK,
        "n_blocks":           N_BLOCKS,
        "feature_order":      ALL_FEATURES,
        "discrete_columns":   DISCRETE_COLUMNS,
        "ctgan_config":       {k: str(v) for k, v in CTGAN_CONFIG.items()},
        "total_time_s":       time.time() - total_start,
        "results":            all_results,
        "log_transform_note": (
            "delta_time log-transformed in Code 2: log1p(x * 1000). "
            "payload_len log-transformed for WILL + Invalid: log1p(x). "
            "Code 4 v4 inverts after generation."
        ),
        "training_note": (
            f"One CTGAN per attack, fit() called {N_BLOCKS} times of "
            f"{EPOCHS_PER_BLOCK} epochs each. Weights are continuous — "
            f"not the old broken loop that reset weights every 10 epochs."
        ),
    }
    with open(OUTPUT_DIR / "training_metadata.json", "w") as fp:
        json.dump(metadata, fp, indent=2)
    mirror("training_metadata.json")

    # Final summary
    print("\n" + "=" * 65)
    print("✅ CODE 3 v5 COMPLETE")
    print("=" * 65)
    print(f"  Total runtime : {elapsed(total_start)}")
    print(f"\n  {'Attack Type':<45} {'Status':<10} {'Blocks':<8} Time")
    print(f"  {'-' * 65}")
    for r in all_results:
        icon = {"OK":"✅","WARNING":"⚠️ ","CRITICAL":"🔴",
                "SKIPPED":"⏭️ ","UNKNOWN":"?"}.get(r.get("status","?"), "?")
        t    = r.get("training_time_s", 0)
        real = sum(1 for c in r.get("block_checks", [])
                   if c["status"] != "SKIPPED")
        print(f"  {icon} {r['attack']:<43} "
              f"{r.get('status','?'):<10} "
              f"{real}/{N_BLOCKS:<6} "
              f"{int(t//60)}m{int(t%60)}s")

    # Advice on any warnings
    bad = [r for r in all_results
           if r.get("status") in ("CRITICAL", "WARNING")]
    if bad:
        print(f"\n  ⚠️  {len(bad)} model(s) need attention:")
        for r in bad:
            print(f"\n     {r['attack']} → {r.get('status')}")
            for c in r.get("block_checks", []):
                if c["status"] not in ("OK", "SKIPPED"):
                    print(f"       Block {c['block']} ({c['epochs']} epochs): "
                          f"{c['status']}  "
                          f"low_var={c['low_var']}  combos={c['combos']}")
        print(f"\n  To extend training by 50 more epochs:")
        print(f"     Set N_BLOCKS = 4")
        resume_hint = {r['attack']: 3 for r in bad}
        print(f"     Set ATTACK_RESUME = {resume_hint}")
        print(f"     Re-run — only block 4 will be trained for flagged attacks")
    else:
        print(f"\n   All models healthy.")

    print(f"\n  Models on Drive : {DRIVE_OUTPUT}")
    print("=" * 65)


if __name__ == "__main__":
    main()
