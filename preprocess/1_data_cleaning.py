import unicodedata, sys

def norm(s):  
    return unicodedata.normalize("NFKC", s.strip())

def split(infile, ja_out, zh_out):
    with open(infile, 'r', encoding='utf-8') as f, \
         open(ja_out, 'w', encoding='utf-8') as wja, \
         open(zh_out, 'w', encoding='utf-8') as wzh:
        for line in f:
            parts = [p.strip() for p in line.split("|||")]
            if len(parts) < 3: continue
            ja, zh = norm(parts[1]), norm(parts[2])
            wja.write(ja + "\n")
            wzh.write(zh + "\n")

split("./data/origin/train.txt", "./data/train.ja", "./data/train.zh")
split("./data/origin/dev.txt",   "./data/valid.ja", "./data/valid.zh")
split("./data/origin/test.txt",  "./data/test.ja",  "./data/test.zh")
print("done")
