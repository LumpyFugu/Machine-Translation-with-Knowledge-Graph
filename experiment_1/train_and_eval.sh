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
USER_DIR="/home/etlpro/XIXI/experiment/experiment_1/src"
DATA=/home/etlpro/XIXI/data/data-bin
SAVE=/home/etlpro/XIXI/experiment/experiment_1/log
SRC=ja
TGT=zh
TB=$SAVE/tb

# Runtime outputs (generation/evaluation intermediates)
#RUN_DIR="$SAVE/run_20260103_170237"
RUN_DIR="${SAVE}/run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
LOG=$RUN_DIR/train.log

# Optional: set GPU
export CUDA_VISIBLE_DEVICES=0

echo "==> USER_DIR : $USER_DIR"
echo "==> DATA     : $DATA"
echo "==> SAVE     : $SAVE"
echo "==> RUN_DIR  : $RUN_DIR"
echo "==> GPU      : $CUDA_VISIBLE_DEVICES"

########################
# Training
########################
fairseq-train "$DATA" \
  --user-dir "$USER_DIR" \
  --task entity_translation \
  --criterion entity_label_smoothed_cross_entropy \
  --arch entity_transformer \
  --label-smoothing 0.1 \
  --loss-gamma 1.0 \
  --src-ner-loss-weight 0.1 --tgt-ner-loss-weight 0.1 \
  --optimizer adam --lr 5e-4 \
  --lr-scheduler inverse_sqrt --warmup-updates 4000 \
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

fairseq-generate "$DATA" \
  --user-dir "$USER_DIR" \
  --task entity_translation \
  --mode 1 \
  --path "$RUN_DIR/checkpoint_best.pt" \
  --source-lang "$SRC" --target-lang "$TGT" \
  --beam 5 --batch-size 8 \
  --remove-bpe=sentencepiece \
  --scoring sacrebleu \
  > "$RUN_DIR/gen.out"

#CUDA_VISIBLE_DEVICES= python /home/etlpro/XIXI/xyc.1/entity_nmt/cli/generate_mode1.py \
#  "$DATA" --user-dir "$USER_DIR" \
#  --task entity_translation --mode 1 \
#  --path "$RUN_DIR/checkpoint_best.pt" \
#  --source-lang "$SRC" --target-lang "$TGT" \
#  --beam 5 --batch-size 1 \
#  --remove-bpe=sentencepiece --scoring sacrebleu \
#  --max-sentences 1 --cpu

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
