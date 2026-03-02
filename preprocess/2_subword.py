import os
import sys
import io
import argparse
import sentencepiece as spm

def ensure_file(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing file: {path}")

def read_lines(path: str):
    with io.open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield line.rstrip("\n")

def write_lines(path: str, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with io.open(path, "w", encoding="utf-8") as w:
        for line in lines:
            w.write(line + "\n")

def train_spm(in_dir: str, out_dir: str, lang: str, vocab_size: int, coverage: float):
    """
    Prefer training SentencePiece on {in_dir}/spm_train.<lang>,
    fall back to {in_dir}/train.<lang> if missing.
    Save model to {out_dir}/spm_<lang>.model / .vocab
    """
    # 1. Check whether spm_train.<lang> exists first
    spm_train_path = os.path.join(in_dir, f"spm_train.{lang}")
    if os.path.isfile(spm_train_path):
        train_path = spm_train_path
        print(f"[{lang.upper()}] Found {spm_train_path}, use it for SPM training.")
    else:
        train_path = os.path.join(in_dir, f"train.{lang}")
        print(f"[{lang.upper()}] Use default {train_path} for SPM training.")

    ensure_file(train_path)

    model_prefix = os.path.join(out_dir, f"spm_{lang}")
    model_path = f"{model_prefix}.model"
    vocab_path = f"{model_prefix}.vocab"

    print(f"[{lang.upper()}] Training SPM on {train_path} (vocab_size={vocab_size}, coverage={coverage})")
    os.makedirs(out_dir, exist_ok=True)
    spm.SentencePieceTrainer.Train(
        input=train_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=coverage,  
        model_type="unigram",
        input_sentence_size=0  
    )
    if not (os.path.exists(model_path) and os.path.exists(vocab_path)):
        raise RuntimeError(f"SPM training failed for {lang}")
    print(f"[{lang.upper()}] Saved: {model_path}, {vocab_path}")

    return model_path
    
    

def encode_splits(in_dir: str, out_dir: str, lang: str, model_path: str):
    """
    Use model_path to encode train/valid/test <lang> into {out_dir}/<split>.<lang>.sp
    Read input from {in_dir}/<split>.<lang>
    """
    sp_proc = spm.SentencePieceProcessor()
    sp_proc.Load(model_path)

    def encode_file(in_file: str, out_file: str):
        print(f"[{lang.upper()}] Encode {os.path.basename(in_file)} -> {os.path.basename(out_file)}")
        lines = read_lines(in_file)
        pieces = (" ".join(sp_proc.EncodeAsPieces(line)) for line in lines)
        write_lines(out_file, pieces)

    for split in ("train", "valid", "test"):
        src = os.path.join(in_dir, f"{split}.{lang}")
        dst = os.path.join(out_dir, f"{split}.{lang}.sp")
        ensure_file(src)
        encode_file(src, dst)

    print(f"[{lang.upper()}] Done.\n")

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess JA & ZH with SentencePiece (input: /home/etl/xyc.1/data, output: /home/etl/xyc.1/data)"
    )
    parser.add_argument("--in-dir",  type=str, default="/home/etl/xyc.1/data",
                        help="Input directory (contains train/valid/test .ja/.zh)")
    parser.add_argument("--out-dir", type=str, default="/home/etl/xyc.1/data",
                        help="Output directory (stores .model/.vocab and *.sp)")
    parser.add_argument("--ja-vocab", type=int, default=64000, help="Japanese SPM vocab size")
    parser.add_argument("--zh-vocab", type=int, default=64000, help="Chinese SPM vocab size")
    parser.add_argument("--ja-coverage", type=float, default=0.9995, help="Japanese coverage")
    parser.add_argument("--zh-coverage", type=float, default=0.9995, help="Chinese coverage")
    args = parser.parse_args()

    in_dir  = os.path.abspath(args.in_dir)
    out_dir = os.path.abspath(args.out_dir)

    if not os.path.isdir(in_dir):
        print(f"Input dir not found: {in_dir}", file=sys.stderr)
        sys.exit(1)
    os.makedirs(out_dir, exist_ok=True)

    print(f"== Input:  {in_dir}")
    print(f"== Output: {out_dir}\n")

    # JA
    ja_model = train_spm(in_dir, out_dir, "ja", args.ja_vocab, args.ja_coverage)
    encode_splits(in_dir, out_dir, "ja", ja_model)

    # ZH
    zh_model = train_spm(in_dir, out_dir, "zh", args.zh_vocab, args.zh_coverage)
    encode_splits(in_dir, out_dir, "zh", zh_model)

    print("All done ✅")

if __name__ == "__main__":
    main()
