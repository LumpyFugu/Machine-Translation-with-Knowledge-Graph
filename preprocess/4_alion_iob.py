# align_iob_spm.py
import sentencepiece as spm
import re

def char_bio_from_word_iob(raw_text, word_tags):
    # Simplified: whitespace tokens + IOB -> char-level BIO (extend B-XXX + I-XXX)
    words = raw_text.strip().split()
    tags  = word_tags.strip().split()
    chars = "".join(words)
    char_tags = []
    for w,t in zip(words, tags):
        L = len(w)
        if t == "O": char_tags += ["O"]*L
        else:
            m = re.match(r"([BI])-([A-Z_]+)", t)
            if not m: char_tags += ["O"]*L
            else:
                typ = m.group(2)
                char_tags += (["B-"+typ] + ["I-"+typ]*(L-1))
    return chars, char_tags

def align_one(sp, raw_text, iob_tags):
    # For robustness, build a no-space char sequence from original text, then segment with SPM and use piece length without '▁' as window size
    _, char_tags = char_bio_from_word_iob(" ".join(list(raw_text.strip())), iob_tags)  # generate BIO per character
    pieces = sp.encode(raw_text.strip(), out_type=str)
    aligned = []
    pos = 0
    for p in pieces:
        plen = len(p.replace("▁",""))
        seg = char_tags[pos:pos+plen] if pos+plen <= len(char_tags) else []
        tag = "O"
        for lab in seg:
            if lab.startswith("B-"): tag = lab; break
            if lab.startswith("I-"): tag = lab
        if tag.startswith("I-") and (len(aligned)==0 or aligned[-1]=="O"):
            tag = "B-"+tag.split("-",1)[1]
        aligned.append(tag)
        pos += plen
    return pieces, aligned

def process(spm_model, txt_in, iob_in, subword_out, iob_out):
    sp = spm.SentencePieceProcessor(model_file=spm_model)
    with open(txt_in, encoding="utf-8") as ft, open(iob_in, encoding="utf-8") as fi, \
         open(subword_out, "w", encoding="utf-8") as fs, open(iob_out, "w", encoding="utf-8") as fo:
        for tline, iline in zip(ft, fi):
            pieces, tags = align_one(sp, tline, iline)
            fs.write(" ".join(pieces)+"\n")
            fo.write(" ".join(tags)+"\n")

# Example: process Ja
process("data/spm_ja.model", "data/train.ja", "data/train.ja.iob", "data/train.ja.sp", "data/train.ja.sp.iob")
process("data/spm_ja.model", "data/valid.ja", "data/valid.ja.iob", "data/valid.ja.sp", "data/valid.ja.sp.iob")
process("data/spm_ja.model", "data/test.ja",  "data/test.ja.iob",  "data/test.ja.sp",  "data/test.ja.sp.iob")
# Example: process Zh
process("data/spm_zh.model", "data/train.zh", "data/train.zh.iob", "data/train.zh.sp", "data/train.zh.sp.iob")
process("data/spm_zh.model", "data/valid.zh", "data/valid.zh.iob", "data/valid.zh.sp", "data/valid.zh.sp.iob")
process("data/spm_zh.model", "data/test.zh",  "data/test.zh.iob",  "data/test.zh.sp",  "data/test.zh.sp.iob")