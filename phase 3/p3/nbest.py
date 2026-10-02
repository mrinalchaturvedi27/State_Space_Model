"""Beam search that returns every finished hypothesis with its raw log-probability.

The search is phase 2's src/evaluate.py:beam_search_decode line for line (single model): the length
penalty only ranks finished hypotheses (score / len**lp), so keeping all of them with raw scores lets
consensus decoding and selective translation re-rank offline at any length penalty.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def beam_search_nbest(model, src, src_key_padding_mask, bos_id, eos_id, pad_id,
                      beam_size=5, max_new_tokens=64):
    """Returns [(token ids incl. bos/eos, raw log-prob)] for every hypothesis the search finished,
    plus the beams still open when max_new_tokens ran out (as beam_search_decode does)."""
    device = src.device
    raw_memory, mem_pad = model.encode(src, src_key_padding_mask)
    memory = raw_memory.expand(beam_size, -1, -1)
    mem_pad = mem_pad.expand(beam_size, -1) if mem_pad is not None else None
    beams = torch.full((beam_size, 1), bos_id, dtype=torch.long, device=device)
    beam_scores = torch.full((beam_size,), float("-inf"), device=device)
    beam_scores[0] = 0.0
    finished = []
    for _ in range(max_new_tokens):
        logits = model.decode(beams, memory, memory_key_padding_mask=mem_pad)
        log_probs = torch.log_softmax(logits[:, -1].float(), dim=-1)
        vocab_size = log_probs.size(-1)
        cand = (beam_scores.unsqueeze(1) + log_probs).view(-1)
        topk_scores, topk_idx = cand.topk(beams.size(0))
        beam_idx = torch.div(topk_idx, vocab_size, rounding_mode="floor")
        tok_idx = topk_idx % vocab_size
        beams = torch.cat([beams[beam_idx], tok_idx.unsqueeze(1)], dim=1)
        beam_scores = topk_scores
        is_eos = tok_idx == eos_id
        if is_eos.any():
            for i in torch.nonzero(is_eos).flatten().tolist():
                finished.append((beams[i].clone(), beam_scores[i].item()))
            keep = ~is_eos
            if keep.sum() == 0:
                beams = beams[:0]
                break
            beams, beam_scores = beams[keep], beam_scores[keep]
            memory = memory[: keep.sum()]
            mem_pad = mem_pad[: keep.sum()] if mem_pad is not None else None
        if beams.size(0) == 0:
            break
    for i in range(beams.size(0)):
        finished.append((beams[i], beam_scores[i].item()))
    if not finished:
        finished.append((torch.tensor([bos_id, eos_id], device=device), 0.0))
    return finished


def rank(finished, lp: float):
    """Finished hypotheses best-first under length penalty lp (ties keep search order)."""
    order = sorted(range(len(finished)), key=lambda i: -finished[i][1] / (finished[i][0].size(0) ** lp))
    return [finished[i] for i in order]
