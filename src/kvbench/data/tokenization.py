from __future__ import annotations


class DummyTokenizer:
    pad_id = 0
    eos_id = 1

    def encode(self, text: str):
        return [ord(c) % 256 for c in text]

    def decode(self, tokens):
        return "".join(chr(t + 48) if isinstance(t, int) else "" for t in tokens)
