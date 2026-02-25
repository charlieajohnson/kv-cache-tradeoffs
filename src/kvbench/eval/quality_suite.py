from __future__ import annotations

from .next_token_acc import next_token_accuracy
from .perplexity import perplexity


def run_quality_suite(logits, targets):
    return {
        "perplexity": perplexity(logits, targets),
        "next_token_accuracy": next_token_accuracy(logits, targets),
    }
