#!/usr/bin/env python3
"""Export the SentencePiece BPE vocab and a golden for the c4tts C++ tokenizer.

The Confucius4-TTS text tokenizer is a LLaMA SentencePiece **BPE** model
(identity normalizer, add_dummy_prefix, byte_fallback). We export id -> (piece,
score, type) so the C++ tokenizer can reproduce spm.encode() bit-exactly.
"""
import glob
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    import sentencepiece as spm
    from sentencepiece import sentencepiece_model_pb2 as pb

    out_dir = os.path.join(REPO_ROOT, "c4tts", "weights", "tokenizer")
    os.makedirs(out_dir, exist_ok=True)
    model = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--netease-youdao--Confucius4-TTS/"
        "snapshots/*/tokenizer.model"))[0]

    mp = pb.ModelProto()
    mp.ParseFromString(open(model, "rb").read())

    # vocab.tsv: one line per id -> "<type>\t<score>\t<piece>" (piece may contain
    # the U+2581 metaspace; bytes are <0xXX> pieces of type 6=BYTE).
    with open(os.path.join(out_dir, "vocab.tsv"), "w", encoding="utf-8") as f:
        for p in mp.pieces:
            piece = p.piece.replace("\t", "\\t").replace("\n", "\\n")
            f.write(f"{int(p.type)}\t{p.score}\t{piece}\n")
    print(f"[tokenizer] wrote {len(mp.pieces)} pieces -> {out_dir}/vocab.tsv")

    # Golden: texts -> the REAL HuggingFace LlamaTokenizer ids (no special
    # tokens), which is what the T2S model was trained/served with. This differs
    # from raw spm.encode in first-token handling (legacy=False: no dummy prefix).
    from transformers import LlamaTokenizer
    snap = os.path.dirname(model)
    tok = LlamaTokenizer.from_pretrained(snap)
    texts = [
        "You are a helpful assistant.",
        "请用中文朗读接下来的文字:你好世界",
        "Hello, world! 123",
        "You are a helpful assistant. 请用中文朗读接下来的文字:你好世界",
        "Mixed café ☕ test — émoji 🎵 end",
        " leading space",
    ]
    gold_dir = os.path.join(REPO_ROOT, "c4tts", "golden", "tokenizer")
    os.makedirs(gold_dir, exist_ok=True)
    with open(os.path.join(gold_dir, "cases.tsv"), "w", encoding="utf-8") as f:
        for t in texts:
            ids = tok.encode(t, add_special_tokens=False)
            f.write(t.replace("\t", " ") + "\t" + ",".join(map(str, ids)) + "\n")
    print(f"[tokenizer] wrote {len(texts)} golden cases -> {gold_dir}/cases.tsv")


if __name__ == "__main__":
    main()
