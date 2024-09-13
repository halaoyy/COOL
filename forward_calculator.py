
from typing import Any
from collections import Counter

import torch

from utils import to_cuda, seq_mask_by_lens


class FinetuneFoward():
    def __init__(self, loss_fn, metrics_fn) -> None:
        self.loss_fn = loss_fn
        self.metrics_fn = metrics_fn

    def compute_forward(self, model, t_batch, s_batch=None, device='cuda:0', cuda:bool = False, evaluate:bool = False, class_balance:bool = False):
        t_input_ids, t_attention_mask, t_entity_embs, t_entity_lens, t_entity_desc_embs, t_entity_neigh_embs, t_labels, t_seq_lens, t_texts_emb = t_batch
        if s_batch is not None:
            s_input_ids, s_attention_mask, s_entity_embs, s_entity_lens, s_entity_desc_embs, s_entity_neigh_embs, s_labels, s_seq_lens, s_texts_emb = s_batch

        if cuda and torch.cuda.is_available():  # type: ignore
            t_input_ids, t_attention_mask, t_entity_embs, t_entity_lens, t_entity_desc_embs, t_entity_neigh_embs, t_labels, t_texts_emb = to_cuda(
                device, data=(t_input_ids, t_attention_mask, t_entity_embs, t_entity_lens, t_entity_desc_embs, t_entity_neigh_embs, t_labels, t_texts_emb))
            if s_batch is not None:
                s_input_ids, s_attention_mask, s_entity_embs, s_entity_lens, s_entity_desc_embs, s_entity_neigh_embs, s_labels, s_texts_emb = to_cuda(
                    device, data=(s_input_ids, s_attention_mask, s_entity_embs, s_entity_lens, s_entity_desc_embs, s_entity_neigh_embs, s_labels, s_texts_emb)
                )
            model = model.to(device)

        if evaluate:
            with torch.no_grad():
                logits = model(t_input_ids, t_attention_mask, t_entity_embs, t_entity_lens, t_entity_desc_embs, t_entity_neigh_embs, t_labels, t_texts_emb)
                return logits
        else:
            logits, loss = model(t_input_ids, t_attention_mask, t_entity_embs, t_entity_lens, t_entity_desc_embs, t_entity_neigh_embs, t_labels, t_texts_emb,
                                 s_input_ids, s_attention_mask, s_entity_embs, s_entity_lens, s_entity_desc_embs, s_entity_neigh_embs, s_labels, s_texts_emb)

        return logits, loss
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.compute_forward(*args, **kwds)
