#!/usr/bin/env python

import argparse
from pathlib import Path

from deeppavlov import build_model, configs


def build_ner_model():
    """
    Build DeepPavlov NER model
    """
    ner_model = build_model(
        configs.ner.ner_ontonotes_bert_mult,
        download=True,
        install=False,
    )
    return ner_model


def read_lines(path):
    """
    Read sys/ref files line by line:
    - normal line: strip and return
    - empty line: replace with a placeholder sentence to avoid empty sequences in NER
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                # placeholder; any symbol is fine, usually no entity inside
                yield "。"
            else:
                yield line


def extract_entities(tokens_batch, tags_batch):
    """
    Extract entities from tokens + BIO tags:
    Return: one list per sentence, each element is a tuple like (label, surface)
    """
    all_entities = []

    for tokens, tags in zip(tokens_batch, tags_batch):
        entities = []
        cur_type = None
        cur_tokens = []

        for tok, tag in zip(tokens, tags):
            if tag.startswith("B-"):
                # close previous entity first
                if cur_type is not None and cur_tokens:
                    surface = "".join(cur_tokens)
                    entities.append((cur_type, surface))
                # start a new entity
                cur_type = tag[2:]
                cur_tokens = [tok]
            elif tag.startswith("I-") and cur_type == tag[2:]:
                cur_tokens.append(tok)
            else:
                # tag == "O" or type mismatch
                if cur_type is not None and cur_tokens:
                    surface = "".join(cur_tokens)
                    entities.append((cur_type, surface))
                cur_type = None
                cur_tokens = []

        # flush at sentence end
        if cur_type is not None and cur_tokens:
            surface = "".join(cur_tokens)
            entities.append((cur_type, surface))

        all_entities.append(entities)

    return all_entities


def run_ner_with_fallback(ner_model, sentences, batch_size, side_name="sys"):
    """
    Run NER on one side (sys or ref):
    1) Call ner_model(batch) in batch_size chunks first
    2) If a batch raises an exception (CRF/vocab/etc.),
       run ner_model([sent]) sentence-by-sentence for that batch,
       and treat persistent failures as no-entity sentences (all O) to avoid exiting the script.
    """
    tokens_all = []
    tags_all = []

    n_problem_batch = 0
    n_problem_sent = 0

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        batch_id = i // batch_size

        try:
            # normal batched call
            batch_tokens, batch_tags = ner_model(batch)
            tokens_all.extend(batch_tokens)
            tags_all.extend(batch_tags)
        except Exception as e:
            # this batch failed, fallback to sentence-by-sentence
            n_problem_batch += 1
            print(f"[WARN][{side_name}] batch {batch_id} "
                  f"(sent idx {i}~{i+len(batch)-1}) raised exception: {repr(e)}")
            print(f"[WARN][{side_name}] Falling back to sentence-by-sentence for this batch.")

            for j, sent in enumerate(batch):
                global_idx = i + j
                try:
                    one_tokens, one_tags = ner_model([sent])
                    tokens_all.extend(one_tokens)
                    tags_all.extend(one_tags)
                except Exception as e2:
                    # if this sentence still fails, treat as no-entity
                    n_problem_sent += 1
                    print(f"[WARN][{side_name}] sentence {global_idx} still raised exception: {repr(e2)}")
                    print(f"[WARN][{side_name}] Treat it as a no-entity sentence. First 50 chars:{repr(sent[:50])}...")

                    # simple fallback: split by char and mark all as "O"
                    chars = list(sent)
                    if not chars:
                        chars = ["。"]
                    tokens_all.append(chars)
                    tags_all.append(["O"] * len(chars))

        if batch_id % 100 == 0:
            print(f"  [{side_name}] processed {i + len(batch)} / {len(sentences)}")

    print(f"[INFO][{side_name}] problem batches: {n_problem_batch}, "
          f"problem sentences (fallback to all-O): {n_problem_sent}")
    return tokens_all, tags_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sys", required=True, help="system output file path, e.g. sys.zh")
    parser.add_argument("--ref", required=True, help="reference file path, e.g. ref.zh")
    parser.add_argument("--out", required=True, help="result output path, e.g. ner_f1.txt")
    args = parser.parse_args()

    sys_path = Path(args.sys)
    ref_path = Path(args.ref)
    out_path = Path(args.out)

    print(f"[INFO] loading NER model ...")
    ner_model = build_ner_model()

    print(f"[INFO] reading sys from {sys_path}")
    sys_sents = list(read_lines(sys_path))
    print(f"[INFO] reading ref from {ref_path}")
    ref_sents = list(read_lines(ref_path))

    assert len(sys_sents) == len(ref_sents), \
        f"sys/ref line count mismatch:{len(sys_sents)} vs {len(ref_sents)}"

    batch_size = 16  # reduce this if memory is insufficient

    print(f"[INFO] running NER on sys in batches of {batch_size} ...")
    sys_tokens_all, sys_tags_all = run_ner_with_fallback(
        ner_model, sys_sents, batch_size, side_name="sys"
    )

    print(f"[INFO] running NER on ref in batches of {batch_size} ...")
    ref_tokens_all, ref_tags_all = run_ner_with_fallback(
        ner_model, ref_sents, batch_size, side_name="ref"
    )

    assert len(sys_tokens_all) == len(ref_tokens_all)

    # Extract entities
    print("[INFO] extracting entities ...")
    gold_entities_per_sent = extract_entities(ref_tokens_all, ref_tags_all)
    pred_entities_per_sent = extract_entities(sys_tokens_all, sys_tags_all)

    tp = fp = fn = 0

    for gold_ents, pred_ents in zip(gold_entities_per_sent, pred_entities_per_sent):
        gold_set = set(gold_ents)  # (type, surface)
        pred_set = set(pred_ents)

        tp += len(gold_set & pred_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as w:
        w.write(f"Entity-level F1 using DeepPavlov NER (treat ref as gold)\n")
        w.write(f"TP={tp}, FP={fp}, FN={fn}\n")
        w.write(f"Precision: {precision:.4f}\n")
        w.write(f"Recall   : {recall:.4f}\n")
        w.write(f"F1       : {f1:.4f}\n")

    print(f"[INFO] done. F1 = {f1:.4f}")
    print(f"[INFO] detailed stats written to: {out_path}")


if __name__ == "__main__":
    main()
