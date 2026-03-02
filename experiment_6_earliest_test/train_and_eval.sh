# Please run the following commands under the `fairseq12` conda environment.
#!/usr/bin/env bash
set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# Training (with BLEU on valid and best-checkpoint saving)
# Final test-set evaluation (fairseq-generate -> sacrebleu for BLEU/chrF)


########################
# Paths & hyperparameters (edit as needed)
########################
# Your custom module (user-dir)
USER_DIR="/home/etlpro/XIXI/experiment/experiment_6_earliest_test/src"
DATA=/home/etlpro/XIXI/data/data-bin
SAVE=/home/etlpro/XIXI/experiment/experiment_6_earliest_test/log
SRC=ja
TGT=zh

############################
# 0) Pin your experiment path (edit here)
############################
EXP="/home/etlpro/XIXI/experiment/experiment_6_earliest_test"
FAIRSEQ_DIR="$EXP/fairseq"   # root directory of your forked fairseq
USER_DIR="$EXP/src"          # your --user-dir

############################
# 1) Explicitly choose which fairseq to use (critical)
############################
export PYTHONPATH="$FAIRSEQ_DIR:${PYTHONPATH:-}"


# Runtime outputs (generation/evaluation intermediates)
RUN_DIR="${SAVE}/run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
TB="$RUN_DIR/tb"
LOG="$RUN_DIR/train.log"
GEN_LOG="$RUN_DIR/gen.out"
mkdir -p "$TB"


# Optional: set GPU
export CUDA_VISIBLE_DEVICES=0

echo "==> USER_DIR : $USER_DIR"
echo "==> DATA     : $DATA"
echo "==> SAVE     : $SAVE"
echo "==> RUN_DIR  : $RUN_DIR"
echo "==> GPU      : $CUDA_VISIBLE_DEVICES"

{
  echo "==================== PROVENANCE ===================="
  echo "[TIME]      $(date -Is)"
  echo "[HOST]      $(hostname)"
  echo "[USER]      $(whoami)"
  echo "[CWD]       $(pwd)"
  echo "[SHELL]     $SHELL"
  echo

  echo "-------------------- BIN PATHS ---------------------"
  echo "[which python]        $(which python || true)"
  echo "[python -V]           $(python -V 2>&1 || true)"
  echo "[which fairseq-train] $(which fairseq-train || true)"
  echo

  echo "-------------------- ENV VARS ----------------------"
  echo "[PYTHONPATH] $PYTHONPATH"
  echo "[CONDA_DEFAULT_ENV] ${CONDA_DEFAULT_ENV:-}"
  echo

  echo "-------------------- FAIRSEQ ORIGIN ----------------"
  python - <<'PY'
import sys
import fairseq
print("[fairseq.__file__] ", fairseq.__file__)
print("[sys.path[0:6]]")
for i,p in enumerate(sys.path[:6]):
    print(f"  {i}: {p}")

import fairseq, fairseq_cli.train, fairseq_cli.generate
print("fairseq:", fairseq.__file__)
print("train_cli:", fairseq_cli.train.__file__)
print("gen_cli:", fairseq_cli.generate.__file__)

from fairseq.data import Dictionary
d = Dictionary.load("/home/etlpro/XIXI/data/data-bin/dict.zh.txt")
print("pad,bos,eos,unk:", d.pad(), d.bos(), d.eos(), d.unk())
# Find the id of underline token '▁' (if it is in vocab)
idx = d.index("▁")
print("'▁' index:", idx)

PY

  echo

  echo "-------------------- PIP METADATA ------------------"
  python -m pip show fairseq 2>/dev/null || echo "[pip show fairseq] (not installed via pip or not found)"
  echo

  echo "================== TRAIN COMMAND ==================="
  printf '%q ' "${TRAIN_CMD[@]}"; echo
  echo "===================================================="
  echo
} | tee -a "$LOG"


########################
# Training
########################
python -m fairseq_cli.train "$DATA" \
  --user-dir "$USER_DIR" \
  --task entity_translation \
  --criterion entity_label_smoothed_cross_entropy \
  --arch entity_transformer \
  --label-smoothing 0.1 \
  --loss-gamma 1.0 \
  --src-ner-loss-weight 0.1 --tgt-ner-loss-weight 0.1 \
  --optimizer adam --lr 2e-4 \
  --lr-scheduler inverse_sqrt --warmup-updates 8000 \
  --max-tokens 4096 \
  --criterion-mode 1 \
  --mode 1 \
  --source-lang "$SRC" --target-lang "$TGT" \
  --tensorboard-logdir "$TB" \
  --log-format simple --log-interval 100 \
  --log-file "$LOG" \
  --max-epoch 20 \
  --patience 5 \
  --ne-dict /home/etlpro/XIXI/data/data-bin/dict.ne.txt \
  --seed 43 \
  --best-checkpoint-metric loss \
  --kg-embed-path /home/etlpro/XIXI/resources/kg_embed.ja.wiki2vec.pt \
  --kg-embed-dim 300 \
  --validate-interval-updates 2000 \
  --save-interval-updates 2000 \
  --keep-interval-updates 5 \
  --save-dir "$RUN_DIR" \
  2>&1 | tee -a "$LOG"

   
 
 #--eval-bleu \
 #--eval-bleu-detok sentencepiece \
 #--eval-bleu-remove-bpe \
 #--eval-bleu-print-samples \
 # --maximize-best-checkpoint-metric \
  

########################
# Generation (inference)
########################
python /home/etlpro/XIXI/experiment/experiment_6_earliest_test/cli/generate_mode1.py "$DATA" \
  --user-dir "$USER_DIR" \
  --task entity_translation \
  --arch entity_transformer \
  --mode 1 \
  --kg-embed-path /home/etlpro/XIXI/resources/kg_embed.ja.wiki2vec.pt \
  --kg-embed-dim 300 \
  --path "$RUN_DIR/checkpoint_best.pt" \
  --source-lang "$SRC" --target-lang "$TGT" \
  --beam 5 --batch-size 64 \
  --remove-bpe=sentencepiece \
  --scoring sacrebleu \
  > "$RUN_DIR/gen.out"


########################
# Extract system outputs and references
########################
# H-: hypothesis (system output), text starts from column 3
grep -P "^H-" "$RUN_DIR/gen.out" | sort -V | cut -f3- > "$RUN_DIR/sys.$TGT"
# T-: reference (dataset target), text starts from column 2
grep -P "^T-" "$RUN_DIR/gen.out" | sort -V | cut -f2- > "$RUN_DIR/ref.$TGT"

########################
# Compute BLEU (sacrebleu)
########################
# For Chinese, -tok zh is recommended; if missing: pip install sacrebleu
if command -v sacrebleu >/dev/null 2>&1; then
  sacrebleu -tok zh "$RUN_DIR/ref.$TGT" < "$RUN_DIR/sys.$TGT" | tee "$RUN_DIR/bleu.txt"
else
  echo "[WARN] sacrebleu is not installed, skipping BLEU. Run: pip install sacrebleu"
fi


echo "==> Full pipeline completed."
echo "Model saved at:       $RUN_DIR"
echo "Generation log:       $RUN_DIR/gen.out"
echo "System output:         $RUN_DIR/sys.$TGT"
echo "Reference:         $RUN_DIR/ref.$TGT"
echo "BLEU (if installed): $RUN_DIR/bleu.txt"


CUDA_VISIBLE_DEVICES="" /home/etlpro/miniconda3/envs/dp-old/bin/python \
  /home/etlpro/XIXI/preprocess/eval_f1.py \
  --sys "$RUN_DIR/sys.$TGT" \
  --ref "$RUN_DIR/ref.$TGT" \
  --out "$RUN_DIR/ner_f1.txt"

echo "NER F1 result:     $RUN_DIR/ner_f1.txt"
