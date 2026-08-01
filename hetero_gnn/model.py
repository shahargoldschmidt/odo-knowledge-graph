"""
HeteroOpioidGNN — 5-node-type heterogeneous GNN for compound-target pKi
regression (HETEROGENEOUS_GNN_ARCHITECTURE.md §6-§8).

Architecture:
  1. Per-node-type encoders (§6): Linear -> BatchNorm -> ReLU -> Dropout(0.3),
     mapping each type's raw vector into a shared 256-dim latent space.
     Assay / Model System / Document raw vectors are themselves built from a
     learned embedding (identity embedding for Assay/ModelSystem; a small
     per-journal embedding + 3 explicit scalars for Document) — see
     encode_nodes() and hetero_gnn/README.md.

  2. 2-hop hub-and-spoke message passing (§7) — NOT a generic multi-relation
     HeteroConv sum. Hop 1 pools Document+ModelSystem+Target into Assay; hop
     2 pools the *updated* Assay into both Compound and Target. Document and
     ModelSystem only ever send messages; they're never themselves updated
     (matches §2.E's "absorb batch effects, not a strong predictor" role).

  3. Edge prediction head (§8): [enriched Compound(256) || enriched
     Target(256) || projected edge_attr] -> MLP -> scalar pKi.
"""
import os
import sys

import torch
import torch.nn as nn
from torch_geometric.utils import scatter
from torch_geometric.utils import softmax

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hetero_gnn.config import (
    ASSAY_EMB_DIM, DOCUMENT_JOURNAL_EMB_DIM, EDGE_PROJ_DIM, ENCODER_DROPOUT,
    HEAD_DROPOUT, HEAD_IN_DIM, HIDDEN, MODEL_SYSTEM_EMB_DIM, TARGET_EMB_DIM,
    TARGET_HIDDEN_IN,
)

BINDS_TO = ("compound", "binds_to", "target")


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _encoder(in_dim: int, hidden: int = HIDDEN, dropout: float = ENCODER_DROPOUT) -> nn.Sequential:
    """§6: Linear -> BatchNorm -> ReLU -> Dropout(0.3), shared by all 5 node types."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
    )


def _hop_matrix(hidden: int = HIDDEN) -> nn.Sequential:
    """§7: 'multiply by a learned weight matrix, apply LeakyReLU', applied to
    [pooled_neighbor_mean || own_vector] (2*hidden -> hidden)."""
    return nn.Sequential(nn.Linear(2 * hidden, hidden), nn.LeakyReLU())


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class HeteroOpioidGNN(nn.Module):
    """
    Parameters
    ----------
    compound_dim         : Compound raw feature dimension (Morgan + numeric ADMET)
    n_targets            : number of unique Target nodes (identity embedding)
    n_assays             : number of unique Assay nodes (identity embedding)
    n_model_systems      : number of unique Model System nodes (identity embedding)
    journal_vocab_size   : Document journal-embedding vocab size (incl. "other")
    edge_dim             : binds_to edge feature dimension
    assay_emb_dim        : Assay identity-embedding size — §5 search range 32-64
    model_system_emb_dim : Model System identity-embedding size — §5 search range 8-24
    document_journal_emb_dim : Document journal-embedding size — §5 search range 4-12
    hidden                : shared latent dimension (§6)
    dropout                : dropout probability (§6/§8)

    The three *_emb_dim args are exactly the dimensions §5 says are chosen by
    hyperparameter search rather than fixed by hand — see search_hparams.py,
    which instantiates this class directly with trial-sampled values instead
    of the config.py defaults used here.
    """

    def __init__(
        self,
        compound_dim: int,
        n_targets: int,
        n_assays: int,
        n_model_systems: int,
        journal_vocab_size: int,
        edge_dim: int,
        assay_emb_dim: int = ASSAY_EMB_DIM,
        model_system_emb_dim: int = MODEL_SYSTEM_EMB_DIM,
        document_journal_emb_dim: int = DOCUMENT_JOURNAL_EMB_DIM,
        hidden: int = HIDDEN,
        dropout: float = ENCODER_DROPOUT,
    ):
        super().__init__()
        self.hidden = hidden

        # --- Node-type encoders (§6) -----------------------------------
        self.compound_encoder = _encoder(compound_dim, hidden, dropout)

        self.target_embedding = nn.Embedding(n_targets, TARGET_EMB_DIM)
        self.target_encoder = _encoder(TARGET_HIDDEN_IN, hidden, dropout)

        self.assay_embedding = nn.Embedding(n_assays, assay_emb_dim)
        self.assay_encoder = _encoder(assay_emb_dim, hidden, dropout)

        self.model_system_embedding = nn.Embedding(n_model_systems, model_system_emb_dim)
        self.model_system_encoder = _encoder(model_system_emb_dim, hidden, dropout)

        self.journal_embedding = nn.Embedding(journal_vocab_size, document_journal_emb_dim)
        self.document_encoder = _encoder(document_journal_emb_dim + 3, hidden, dropout)  # + is_patent, norm_year, is_unknown

        # --- Message passing (§7): hop 1 -> Assay; hop 2 -> Compound & Target
        self.W1 = _hop_matrix(hidden)
        
        # New Attention layers for scoring neighbors
        self.attn_hop1 = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1))
        self.attn_hop2_c = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1))
        self.attn_hop2_t = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1))

        self.W2_compound = _hop_matrix(hidden)
        self.W2_target = _hop_matrix(hidden)
        
        # --- Prediction head (§8) ---------------------------------------
        self.edge_proj = nn.Linear(edge_dim, EDGE_PROJ_DIM)   # "narrow linear layer"
        self.edge_head = nn.Sequential(
            nn.Linear(HEAD_IN_DIM, 256), nn.ReLU(), nn.Dropout(HEAD_DROPOUT),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )
        

    # -----------------------------------------------------------------------
    # §6 — encode every node type's raw vector into the shared 256-dim space
    # -----------------------------------------------------------------------

    def encode_nodes(self, data):
        h_c = self.compound_encoder(data["compound"].x)

        tgt_emb = self.target_embedding(data["target"].node_idx)
        h_t = self.target_encoder(torch.cat([data["target"].x, tgt_emb], dim=-1))

        assay_emb = self.assay_embedding(data["assay"].node_idx)
        h_a = self.assay_encoder(assay_emb)

        ms_emb = self.model_system_embedding(data["model_system"].node_idx)
        h_m = self.model_system_encoder(ms_emb)

        journal_emb = self.journal_embedding(data["document"].journal_idx)
        h_d = self.document_encoder(torch.cat([journal_emb, data["document"].x], dim=-1))

        return h_c, h_t, h_a, h_m, h_d

    # -----------------------------------------------------------------------
    # §7 — 2-hop hub-and-spoke message passing
    # -----------------------------------------------------------------------

    def message_pass(self, h_c, h_t, h_a, h_m, h_d, data):
        n_assays = h_a.shape[0]

        # --- Hop 1: pool {Document, ModelSystem, Target} into Assay ------
        src_d, dst_d = data[("document", "rev_has_document", "assay")].edge_index
        src_m, dst_m = data[("model_system", "rev_has_model_system", "assay")].edge_index
        src_t, dst_t = data[("target", "rev_tests_target", "assay")].edge_index

        neighbor_vecs = torch.cat([h_d[src_d], h_m[src_m], h_t[src_t]], dim=0)
        neighbor_dst = torch.cat([dst_d, dst_m, dst_t], dim=0)
        
        # Attention logic instead of mean:
        attn_scores_1 = self.attn_hop1(neighbor_vecs) # [num_neighbors, 1]
        attn_weights_1 = softmax(attn_scores_1, index=neighbor_dst, dim=0, num_nodes=n_assays)
        weighted_neighbors_1 = neighbor_vecs * attn_weights_1
        
        pooled_a = scatter(weighted_neighbors_1, neighbor_dst, dim=0, dim_size=n_assays, reduce="sum")
        h_a_new = self.W1(torch.cat([pooled_a, h_a], dim=-1))

        # --- Hop 2: pool the *updated* Assay into Compound and Target ----
        src_ac, dst_c = data[("assay", "rev_tested_in", "compound")].edge_index
        
        attn_scores_c = self.attn_hop2_c(h_a_new[src_ac])
        attn_weights_c = softmax(attn_scores_c, index=dst_c, dim=0, num_nodes=h_c.shape[0])
        weighted_neighbors_c = h_a_new[src_ac] * attn_weights_c
        
        pooled_c = scatter(weighted_neighbors_c, dst_c, dim=0, dim_size=h_c.shape[0], reduce="sum")
        h_c_new = self.W2_compound(torch.cat([pooled_c, h_c], dim=-1))

        src_at, dst_t2 = data[("assay", "tests_target", "target")].edge_index
        
        attn_scores_t = self.attn_hop2_t(h_a_new[src_at])
        attn_weights_t = softmax(attn_scores_t, index=dst_t2, dim=0, num_nodes=h_t.shape[0])
        weighted_neighbors_t = h_a_new[src_at] * attn_weights_t
        
        pooled_t = scatter(weighted_neighbors_t, dst_t2, dim=0, dim_size=h_t.shape[0], reduce="sum")
        h_t_new = self.W2_target(torch.cat([pooled_t, h_t], dim=-1))

        return h_c_new, h_t_new

    # -----------------------------------------------------------------------
    # §8 — prediction head
    # -----------------------------------------------------------------------

    def predict_edges(self, h_c, h_t, edge_index, edge_attr):
        src = h_c[edge_index[0]]
        dst = h_t[edge_index[1]]
        e_proj = self.edge_proj(edge_attr)
        e = torch.cat([src, dst, e_proj], dim=-1)
        return self.edge_head(e).squeeze(-1)

    def forward(self, data, edge_index=None, edge_attr=None):
        h_c, h_t, h_a, h_m, h_d = self.encode_nodes(data)
        h_c, h_t = self.message_pass(h_c, h_t, h_a, h_m, h_d, data)

        if edge_index is None:
            edge_index = data[BINDS_TO].edge_index
            edge_attr = data[BINDS_TO].edge_attr

        return self.predict_edges(h_c, h_t, edge_index, edge_attr)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(
    data,
    edge_dim: int,
    assay_emb_dim: int = ASSAY_EMB_DIM,
    model_system_emb_dim: int = MODEL_SYSTEM_EMB_DIM,
    document_journal_emb_dim: int = DOCUMENT_JOURNAL_EMB_DIM,
) -> HeteroOpioidGNN:
    model = HeteroOpioidGNN(
        compound_dim=data["compound"].x.shape[1],
        n_targets=data["target"].num_nodes,
        n_assays=data["assay"].num_nodes,
        n_model_systems=data["model_system"].num_nodes,
        journal_vocab_size=data["document"].journal_vocab_size,
        edge_dim=edge_dim,
        assay_emb_dim=assay_emb_dim,
        model_system_emb_dim=model_system_emb_dim,
        document_journal_emb_dim=document_journal_emb_dim,
    )

    # Shift output bias to the training-set mean pKi so the model starts near
    # the correct value range from epoch 1 (weights stay at default init).
    # Uses exact-labelled train edges only (data must come from
    # dataset.build_dataset(), which sets train_exact_mask) — hinge-bound
    # edges are one-sided censored values, not true means, and would skew
    # this initialisation.
    train_mean = float(data[BINDS_TO].edge_label[data[BINDS_TO].train_exact_mask].mean())
    model.edge_head[-1].bias.data.fill_(train_mean)

    return model
