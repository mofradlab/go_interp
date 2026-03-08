#!/usr/bin/env bash
# run_eval.sh — Regenerate all eval_files/ outputs from scratch.
#
# Run from go_ml/dataset_eval/. Each command produces pkl files under
# eval_files/{dataset}/ that are consumed by eval_models.ipynb.
#
# Approximate runtimes per command on a single A100 GPU:
#   cond_bert_gen_esmc.py  ~2 h per param_index (5 runs total = ~10 h)
#   Other notebooks        ~30 min each
#
# Usage:
#   bash run_eval.sh [GPU_ID]   # defaults to GPU 0

set -euo pipefail

GPU=${1:-0}

echo "=== ESMC conditioned models (5 hyperparameter configurations) ==="
for IDX in 0 1 2 3 4; do
    echo "--- param_index=$IDX ---"
    python cond_bert_gen_esmc.py --param_index "$IDX" --gpu_id "$GPU"
done

echo ""
echo "=== Remaining baselines ==="
echo "Run the following notebooks (from go_ml/dataset_eval/) in order:"
echo ""
echo "  1. cond_bert_gen_esm2.ipynb   — ESM2-conditioned + unconditioned ESMC baseline"
echo "     Produces: esm_cond.pkl, esm_cond_span.pkl, esmc.pkl, esmc_cond_span.pkl"
echo ""
echo "  2. bert_gen_esmfast.ipynb     — ESM-Fast and ESM2 unconditioned baselines"
echo "     Produces: esm_fast.pkl, esm_esm2.pkl, esm_fast_esm2.pkl"
echo ""
echo "  3. bert_gen.ipynb             — Basic BERT (ESMC unconditioned) baseline"
echo "     Produces: esmc.pkl (if not already done above)"
echo ""
echo "  4. annot_bert_gen.ipynb       — ProteinInfer score prediction baseline"
echo "     Produces: score_pred.pkl"
echo ""
echo "  5. msa_gen.ipynb              — MSA PSSM entropy baseline"
echo "     Produces: msa.pkl  (requires MSA pipeline to have been run first)"
echo ""
echo "  6. ../model_interp/gen_interp.ipynb      — Attribution methods"
echo "     Produces: lga_attr.pkl, ldl_attr.pkl (paper), plus lgs_attr, lig_attr, lc_attr, ig_attr"
echo ""
echo "  7. ../model_interp/attention_interp.ipynb — Attention attribution"
echo "     Produces: esmc_attention.pkl"
echo ""
echo "All outputs verified: run eval_models.ipynb top-to-bottom to reproduce results."
