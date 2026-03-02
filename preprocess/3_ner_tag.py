# Please run the following commands under the `dp-old` conda environment.
import os
from deeppavlov import build_model

# === Config ===
DATA_DIR = "/home/etl/xyc.1/data"
BATCH = 64
INPUT_LANGS = ["ja", "zh"]
SPLITS = ["train", "valid", "test"]

# [Added] Chunk params (character-based) — tune for machine/data
CHUNK_CHARS = 200     # Max chars per chunk (conservative for CJK text)
CHUNK_OVERLAP = 30    # Overlap chars between adjacent chunks to avoid splitting entities
# ==============

model = build_model("ner_ontonotes_bert_mult", download=True)

def read_lines(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield line.rstrip("\n")

def write_lines(path, lines):
    with open(path, "w", encoding="utf-8") as w:
        for ln in lines:
            w.write(ln + "\n")

# [Added] Sliding-window chunking by characters (with overlap)
def chunk_by_chars(text, max_chars=CHUNK_CHARS, overlap=CHUNK_OVERLAP):
    if len(text) <= max_chars:
        return [text]
    chunks = []
    i = 0
    while i < len(text):
        j = min(i + max_chars, len(text))
        chunks.append(text[i:j])
        if j == len(text):
            break
        i = j - overlap  # Step back by overlap chars
        if i < 0:
            i = 0
    return chunks

def infer_batch(sent_batch):
    out = model(sent_batch)
    # DeepPavlov has two common return formats:
    # A) tags_list
    if out and isinstance(out, list) and isinstance(out[0], list) and \
       (len(out[0]) == 0 or isinstance(out[0][0], str)):
        tags_batch = out
    # B) [tokens_list, tags_list]
    elif isinstance(out, (list, tuple)) and len(out) == 2:
        tags_batch = out[1]
    else:
        raise RuntimeError(f"Unexpected model output structure: type={type(out)}")
    return tags_batch

def tag_one_sentence(sent):
    # Return empty for empty lines
    if not sent.strip():
        return ""

    # [Added] Chunk by chars first, run NER chunk-by-chunk, then concatenate tags
    pieces = chunk_by_chars(sent)
    all_tags = []
    # For robustness, process chunk-by-chunk (chunks are small enough)
    for piece in pieces:
        tags_piece = infer_batch([piece])[0]  # returns tag sequence for one sentence
        all_tags.extend(tags_piece)

    # Output one line: space-separated IOB tags
    return " ".join(all_tags)

def tag_file(inp_path, out_path):
    if not os.path.exists(inp_path):
        raise FileNotFoundError(f"Input not found: {inp_path}")

    print(f"[INFO] Tagging: {inp_path} -> {out_path}")
    outputs, cnt = [], 0
    for sent in read_lines(inp_path):
        outputs.append(tag_one_sentence(sent))
        cnt += 1
        if cnt % 1000 == 0:
            print(f"  processed {cnt} lines...")
    write_lines(out_path, outputs)
    print(f"[DONE] {inp_path} -> {out_path}, lines={len(outputs)}")

def main():
    for s in SPLITS:
        for lang in INPUT_LANGS:
            inp = os.path.join(DATA_DIR, f"{s}.{lang}")
            outp = os.path.join(DATA_DIR, f"{s}.{lang}.iob")
            tag_file(inp, outp)
    print("All done.")

if __name__ == "__main__":
    main()
