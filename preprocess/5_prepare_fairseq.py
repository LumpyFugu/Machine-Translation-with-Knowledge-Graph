import os
import shutil
import subprocess
from pathlib import Path


DATA_DIR = Path("~/XIXI/data/data")
DEST_DIR = Path("~/XIXI/data/data-bin")

def main():
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    # Plain text
    for split in ["train", "valid", "test"]:
        src_file = DATA_DIR / f"{split}.ja.sp"
        tgt_file = DATA_DIR / f"{split}.zh.sp"
        out_ja = DATA_DIR / f"{split}.ja-zh.ja"
        out_zh = DATA_DIR / f"{split}.ja-zh.zh"

        if not src_file.exists() or not tgt_file.exists():
            print(f"[Error] Missing file: {src_file} or {tgt_file}")
            return
        
        shutil.copy(src_file, out_ja)
        shutil.copy(tgt_file, out_zh)

    # Call fairseq-preprocess (text)
    subprocess.run([
        "fairseq-preprocess",
        "--source-lang", "ja",
        "--target-lang", "zh",
        "--trainpref", str(DATA_DIR / "train.ja-zh"),
        "--validpref", str(DATA_DIR / "valid.ja-zh"),
        "--testpref", str(DATA_DIR / "test.ja-zh"),
        "--destdir", str(DEST_DIR)
    ], check=True)

    # Entity data
    for split in ["train", "valid", "test"]:
        src_iob = DATA_DIR / f"{split}.ja.sp.iob"
        tgt_iob = DATA_DIR / f"{split}.zh.sp.iob"
        out_ja_ne = DATA_DIR / f"{split}.ja.ne-zh.ne.ja.ne"
        out_zh_ne = DATA_DIR / f"{split}.ja.ne-zh.ne.zh.ne"

        if not src_iob.exists() or not tgt_iob.exists():
            print(f"[Error] Missing entity file: {src_iob} or {tgt_iob}")
            return
        
        shutil.copy(src_iob, out_ja_ne)
        shutil.copy(tgt_iob, out_zh_ne)

    subprocess.run([
        "fairseq-preprocess",
        "--source-lang", "ja.ne",
        "--target-lang", "zh.ne",
        "--trainpref", str(DATA_DIR / "train.ja.ne-zh.ne"),
        "--validpref", str(DATA_DIR / "valid.ja.ne-zh.ne"),
        "--testpref", str(DATA_DIR / "test.ja.ne-zh.ne"),
        "--destdir", str(DEST_DIR)
    ], check=True)

if __name__ == "__main__":
    main()
