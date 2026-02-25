from __future__ import annotations

from kvbench.data.tokenization import DummyTokenizer


def main():
    tok = DummyTokenizer()
    print("vocab_size", 256)
    print("pad_id", tok.pad_id)


if __name__ == "__main__":
    main()
