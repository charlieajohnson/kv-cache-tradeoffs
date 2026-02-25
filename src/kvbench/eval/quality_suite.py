from __future__ import annotations

from .perplexity import perplexity
from .next_token_acc import next_token_accuracy


def run_quality_suite(logits, targets):
    return {
        "perplexity": perplexity(logits, targets),
        "next_token_accuracy": next_token_accuracy(logits, targets),
    }
