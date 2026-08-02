"""Central configuration for the heterogeneous 5-node-type ODO GNN.

Every constant that resolves a "TBD" / "Open Decision" in
HETEROGENEOUS_GNN_ARCHITECTURE.md is set here, with a comment pointing back
to the relevant section. See hetero_gnn/README.md for the full rationale.
"""
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG_DIR     = os.path.dirname(os.path.abspath(__file__))
EXCEL_PATH  = os.path.join(ROOT, "Final ODO Dataset_v2026-06-10.xlsx")
GRAPH_PATH  = os.path.join(PKG_DIR, "processed_hetero_graph.pt")
COMPOUND_BUILDER_PATH = os.path.join(PKG_DIR, "compound_feature_builder.pt")
TARGET_REFERENCE_PATH = os.path.join(PKG_DIR, "target_reference.csv")

CHECKPOINT_DIR = os.path.join(PKG_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Compound features (§2.A, §5)  — Morgan(2048) + 12 explicit numeric columns
# ---------------------------------------------------------------------------
MORGAN_BITS   = 2048
MORGAN_RADIUS = 2

# Exact columns listed under Compound §2.A "PK (ChEMBL)" + "Computed (QikProp)".
# 2 + 10 = 12 numeric columns -> Compound dim = 2048 + 12 = 2060 (doc's "~2,058"
# is an approximation; the explicit §2.A column list is the more precise source
# of truth and is what's implemented here).
COMPOUND_NUMERIC_COLS = [
    "chembl_alogp", "chembl_#ro5_violations",
    "qikprop_dipole", "qikprop_sasa", "qikprop_fisa",
    "qikprop_donor_hb", "qikprop_accpt_hb",
    "qikprop_qplog_pw", "qikprop_qplog_po/w", "qikprop_qplogs",
    "qikprop_qplog_khsa", "qikprop_percent_human_oral_absorption",
]

# ---------------------------------------------------------------------------
# Target features (§2.B, §5) — 4 (type) + 9 (taxonomy) one-hot, exact-sized,
# no catch-all slot (unmatched/missing -> all-zero). + 32-dim learned
# embedding indexed by uniprot_protein_id (the node's own primary key).
# Vocabularies are built from the *actual* data (see preprocess.py), capped
# to these sizes by descending frequency, not hardcoded.
# ---------------------------------------------------------------------------
N_TARGET_TYPE_CATS = 4
N_TARGET_TAXONOMY_CATS = 9
TARGET_EMB_DIM = 32
TARGET_HIDDEN_IN = N_TARGET_TYPE_CATS + N_TARGET_TAXONOMY_CATS + TARGET_EMB_DIM  # 45

# ---------------------------------------------------------------------------
# Assay / Model System / Document embedding dims (§5): "not fixed by hand —
# they're chosen by hyperparameter search ... selecting the value with the
# best validation loss". The *_EMB_DIM_RANGE tuples are the doc's exact
# search bounds, used by search_hparams.py. The *_EMB_DIM point constants
# below are the current default (used by a plain run_train.py run, or as a
# starting point until you run the search and adopt its winner via
# run_train.py's --assay-emb-dim / --model-system-emb-dim /
# --document-journal-emb-dim flags) — they're the midpoint of each range.
#
# All three are still *pure learned identity embeddings* (Assay/ModelSystem)
# or a small per-journal embedding + 3 scalars (Document) — that part of the
# design is unchanged from before; only the exact dimension is now
# search-selected rather than hand-picked. See hetero_gnn/README.md.
# ---------------------------------------------------------------------------
ASSAY_EMB_DIM_RANGE = (32, 64)
MODEL_SYSTEM_EMB_DIM_RANGE = (8, 24)
DOCUMENT_JOURNAL_EMB_DIM_RANGE = (4, 12)

ASSAY_EMB_DIM = 48
MODEL_SYSTEM_EMB_DIM = 16
DOCUMENT_JOURNAL_EMB_DIM = 8
DOCUMENT_HIDDEN_IN = DOCUMENT_JOURNAL_EMB_DIM + 3  # + is_patent + norm_year + unknown_document_flag

# ---------------------------------------------------------------------------
# Edge (`binds_to`) features (§3, §5) — 7 + ~4 + ~3 = ~14-16, exact in doc.
# ---------------------------------------------------------------------------
# §3 literal enumeration incl. the explicit "other" catch-all -> 7 dims.
ENDPOINT_TYPES = ["Ki", "IC50", "EC50", "Inhibition", "Activity", "Binding", "other"]

# "~4": real data qualifier vocab is dominated by =, <, > (>=99.9%); <=, >=,
# ~ and the #ERROR! / unrecoverable bucket are all rare (<0.1% combined) and
# share the same fate downstream (never loss-eligible - loss_mask requires
# an exact '=' match), so they're bucketed into one "other" slot -> 4 dims.
QUALIFIERS = ["=", "<", ">", "other"]

# "~3" for pharmacological role, but chembl_binding_site_description is also
# an explicit §3 column, so it gets its own small one-hot too:
#   pharm role (3: agonist / antagonist / other-or-missing)
# + binding site (2: high_affinity / low_affinity, missing -> all-zero)
# = 5 dims. Total edge dim = 7 + 4 + 5 = 16 (the top of the doc's "~14-16").
PHARM_ROLES = ["agonist", "antagonist", "other"]
BINDING_SITES = ["high_affinity", "low_affinity"]

EDGE_ATTR_DIM = len(ENDPOINT_TYPES) + len(QUALIFIERS) + len(PHARM_ROLES) + len(BINDING_SITES)  # 16

# ---------------------------------------------------------------------------
# pKi derivation (§1) — used only when pchembl_value is missing.
# Only pure molar-concentration units are convertible; '%', 'mg kg-1', etc.
# cannot be turned into a Ki-like concentration and are left unlabelled.
# ---------------------------------------------------------------------------
UNIT_TO_MOLAR = {"nM": 1e-9, "uM": 1e-6}

# §1: loss has two components, both restricted to endpoint == 'Ki' with a
# resolvable qualifier (never '#ERROR!'/unrecoverable, never '~' — the doc
# gives no hinge direction for "approximately equal" and it's a single row
# in the whole dataset, so it's treated as unrecoverable-for-loss too).
LOSS_ENDPOINT = "Ki"

# Exact loss (MSE): qualifier == '=' AND label in [MIN_PCHEMBL, MAX_PCHEMBL].
LOSS_QUALIFIER = "="
MIN_PCHEMBL = 2.0
MAX_PCHEMBL = 15.0

# Censored loss (one-sided hinge), branched on the *raw* qualifier (§1.2):
# raw '<'/'<=' -> the derived/looked-up pKi is a FLOOR (true pKi is higher);
# raw '>'/'>=' -> it's a CEILING (true pKi is lower). '<=' and '>=' aren't
# named explicitly in the doc's two worked examples but carry the same
# inequality direction as '<'/'>' (just inclusive), so they get the same
# treatment. No MIN/MAX_PCHEMBL range clamp is applied here — the doc states
# that guard only for the exact-loss bullet, not the hinge one.
HINGE_FLOOR_QUALIFIERS = {"<", "<="}
HINGE_CEILING_QUALIFIERS = {">", ">="}
CENSORED_LOSS_WEIGHT = 1.0   # doc: "tune empirically" — no number given, so left at parity with exact loss

# ---------------------------------------------------------------------------
# Shared latent space + encoders (§6)
# ---------------------------------------------------------------------------
HIDDEN = 256
ENCODER_DROPOUT = 0.5   # doc §6 is explicit: Linear -> BatchNorm -> ReLU -> Dropout(0.4)

# ---------------------------------------------------------------------------
# Message passing (§7) — capped at 2 hops, custom hub-and-spoke scheme
# (not a generic multi-relation HeteroConv sum). See model.py.
# ---------------------------------------------------------------------------
# (no extra config needed beyond HIDDEN — W1/W2/W2_target are all
#  Linear(2*HIDDEN, HIDDEN) inside the model)

# ---------------------------------------------------------------------------
# Prediction head (§8)
# ---------------------------------------------------------------------------
# Edge vector (16) passed through a "narrow linear layer" before
# concatenation. Sized so HIDDEN + HIDDEN + EDGE_PROJ_DIM == 530 (doc's
# literal "~530" target), i.e. 256 + 256 + 18 = 530.
EDGE_PROJ_DIM = 18
HEAD_IN_DIM = HIDDEN + HIDDEN + EDGE_PROJ_DIM  # 530
HEAD_DROPOUT = ENCODER_DROPOUT  # doc gives one explicit dropout figure (§6); reused for the head's Dropout (§8)

# ---------------------------------------------------------------------------
# Temporal split (§9, Open Decision #1 — TBD, resolved here)
# ---------------------------------------------------------------------------
# Reuses the cutoff already validated for the bipartite model (gnn/config.py):
# papers up to 2014 -> train/val, 2015 onward -> held-out test. No upper cap
# (the older TEST_MAX_YEAR=2020 convention in preprocess_bipartite_graph.py
# is superseded by gnn/dataset.py's open-ended cutoff, which this follows).
TEMPORAL_CUTOFF_YEAR = 2015     # train: year < cutoff : test: year >= cutoff
VAL_FRAC_OF_TRAIN = 0.10        # random 10% of pre-cutoff edges held out as val

# ---------------------------------------------------------------------------
# Optimization (§9)
# ---------------------------------------------------------------------------
SEED          = 42
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
MAX_EPOCHS    = 200
PATIENCE      = 20      # early stopping patience (epochs without val improvement)
LR_PATIENCE   = 10      # ReduceLROnPlateau patience
BATCH_SIZE    = 512     # binds_to edges per mini-batch
