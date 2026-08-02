"""
search_hparams.py
──────────────────
§5: "Assay, Model System, and Document dimensions are not fixed by hand —
they're chosen by hyperparameter search (grid/random search across training
iterations, selecting the value with the best validation loss), same as any
other tunable hyperparameter in this pipeline."

Optuna, TPE sampler + MedianPruner, searching exactly the doc's stated
ranges (§5):
  assay_emb_dim             in [32, 64]
  model_system_emb_dim      in [8, 24]
  document_journal_emb_dim  in [4, 12]

A full MAX_EPOCHS (200, early-stopped) run per trial would be prohibitively
expensive for a multi-trial search, so each trial trains with a reduced
epoch budget + tighter early-stopping patience, reporting intermediate
validation RMSE (exact-labelled edges only — see train.py's module
docstring for why) every epoch so the pruner can kill clearly-unpromising
trials early. The objective Optuna minimizes is the best val RMSE reached
within that trial's budget.

After the search, the winning (assay/model_system/document) combination is
saved to hetero_gnn/best_hparams.json and — unless --no-final-run is
passed — immediately used for one real, full-budget training run (the same
train.train() used by run_train.py), so checkpoints/best_model.pt ends up
holding the best model actually found, ready for predict.py.

Usage:
    conda run -n odo python3 hetero_gnn/search_hparams.py --n-trials 20
    conda run -n odo python3 hetero_gnn/search_hparams.py --n-trials 20 --no-final-run
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optuna
import torch

from hetero_gnn.config import (
    ASSAY_EMB_DIM_RANGE, DOCUMENT_JOURNAL_EMB_DIM_RANGE, LR, MODEL_SYSTEM_EMB_DIM_RANGE,
    PKG_DIR, SEED, WEIGHT_DECAY,
)
from hetero_gnn.dataset import build_dataset, BINDS_TO
from hetero_gnn.model import build_model
from hetero_gnn.train import evaluate, train, train_epoch

SEARCH_MAX_EPOCHS = 40   # reduced budget per trial (vs MAX_EPOCHS=200 for the final run)
SEARCH_PATIENCE = 8      # tighter early-stopping within a trial
BEST_CONFIG_PATH = os.path.join(PKG_DIR, "best_hparams.json")


def _run_trial(trial: optuna.Trial, data, device, search_epochs: int, search_patience: int) -> float:
    assay_emb_dim = trial.suggest_int("assay_emb_dim", *ASSAY_EMB_DIM_RANGE)
    model_system_emb_dim = trial.suggest_int("model_system_emb_dim", *MODEL_SYSTEM_EMB_DIM_RANGE)
    document_journal_emb_dim = trial.suggest_int("document_journal_emb_dim", *DOCUMENT_JOURNAL_EMB_DIM_RANGE)

    model = build_model(
        data, edge_dim=data[BINDS_TO].edge_attr.shape[1],
        assay_emb_dim=assay_emb_dim,
        model_system_emb_dim=model_system_emb_dim,
        document_journal_emb_dim=document_journal_emb_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_rmse = float("inf")
    no_improve = 0
    for epoch in range(1, search_epochs + 1):
        train_epoch(model, data, optimizer, device)
        rmse = evaluate(model, data, "val", device)["rmse"]

        trial.report(rmse, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if rmse < best_rmse:
            best_rmse = rmse
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= search_patience:
            break

    return best_rmse


def main():
    parser = argparse.ArgumentParser(
        description="§5 hyperparameter search over Assay/Model System/Document embedding dims."
    )
    parser.add_argument("--n-trials", type=int, default=20, help="Number of Optuna trials (default: 20).")
    parser.add_argument("--search-epochs", type=int, default=SEARCH_MAX_EPOCHS,
                         help=f"Reduced epoch budget per trial (default: {SEARCH_MAX_EPOCHS}).")
    parser.add_argument("--search-patience", type=int, default=SEARCH_PATIENCE,
                         help=f"Early-stopping patience within a trial (default: {SEARCH_PATIENCE}).")
    parser.add_argument("--no-final-run", action="store_true",
                         help="Skip the full-budget final training run with the winning config.")
    parser.add_argument("--seed", type=int, default=SEED, help="Optuna TPE sampler seed.")
    args = parser.parse_args()

    print("=" * 60)
    print("  §5 Hyperparameter Search — Assay / Model System / Document dims")
    print(f"  Ranges: assay={ASSAY_EMB_DIM_RANGE}  model_system={MODEL_SYSTEM_EMB_DIM_RANGE}  "
          f"document_journal={DOCUMENT_JOURNAL_EMB_DIM_RANGE}")
    print(f"  {args.n_trials} trials, up to {args.search_epochs} epochs each "
          f"(patience {args.search_patience}), TPE sampler + median pruner")
    print("=" * 60)

    data, meta = build_dataset(verbose=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    data = data.to(device)

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=15)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(
        lambda trial: _run_trial(trial, data, device, args.search_epochs, args.search_patience),
        n_trials=args.n_trials,
    )

    print("\n" + "=" * 60)
    print(f"  Best val RMSE: {study.best_value:.4f}")
    print(f"  Best params  : {study.best_params}")
    n_pruned = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
    n_complete = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    print(f"  {n_complete} completed, {n_pruned} pruned, {len(study.trials)} total")
    print("=" * 60)

    with open(BEST_CONFIG_PATH, "w") as f:
        json.dump({"best_val_rmse": study.best_value, **study.best_params}, f, indent=2)
    print(f"\n  Saved → {BEST_CONFIG_PATH}")
    p = study.best_params
    print(
        "  To reproduce with the full training script:\n"
        f"    conda run -n odo python3 hetero_gnn/run_train.py "
        f"--assay-emb-dim {p['assay_emb_dim']} "
        f"--model-system-emb-dim {p['model_system_emb_dim']} "
        f"--document-journal-emb-dim {p['document_journal_emb_dim']}"
    )

    if args.no_final_run:
        return

    print("\n" + "=" * 60)
    print("  Final full-budget run with the winning hyperparameters …")
    print("=" * 60)
    model = build_model(
        data, edge_dim=data[BINDS_TO].edge_attr.shape[1],
        assay_emb_dim=p["assay_emb_dim"],
        model_system_emb_dim=p["model_system_emb_dim"],
        document_journal_emb_dim=p["document_journal_emb_dim"],
    )
    train(model, data, verbose=True)
    print("\nDone. checkpoints/best_model.pt now holds the searched-and-trained best model.")


if __name__ == "__main__":
    main()
