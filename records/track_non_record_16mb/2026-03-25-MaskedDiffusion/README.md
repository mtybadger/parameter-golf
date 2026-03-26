This is a non-record submission that replaces the autoregressive `train_gpt.py` baseline with a masked diffusion language model implemented in`train_mdlm.py`. The MDLM is from ["Simple and Effective Masked Diffusion Language Models"](https://arxiv.org/pdf/2406.07524), and the code inspired by ["that paper's repo"](https://github.com/kuleshov-group/mdlm)

The model keeps much of the original training harness and systems stack from the original baseline, but swaps the causal next-token objective for a bidirectional masked denoising objective with sigma conditioning and iterative sampling.

## Config
- Tokenizer/data: reuses FineWeb SP-1024, one extra \[MASK\] token added for 1025 vocab size
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=8 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`; 8 layers needed since conditioning adds extra params. Arguably not a completely fair comparison with the baseline due to the missing layer.
- Dropout 0: Arguably less important in diffusion models since they already handle a lot of noise, which acts as regularization in the same way dropout does.
- Attention: bidirectional transformer with GQA-style `NUM_KV_HEADS=4`, with adaLN conditioning
- Conditioning: timestep-conditioned denoiser with reduced internal conditioning width `cond_dim=max(model_dim//4, 64)`
- Batch/sequence defaults: `TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=256`. Lower sequence length because it's a bidirectional model
- Sampling defaults: `SAMPLER=ddpm_cache SAMPLING_SCHEDULE=linear SAMPLING_STEPS=256`. N.b. probably lots of fun to be had with the sampling schedule!
- Variational eval: `VAR_EVAL_STEPS=64`. I'm interested in whether using more val steps gives better performance, which would be a kind of test-time compute native to diffusion models! Validation takes 

## Metrics
- `val_loss` is the continuous-time SUBS denoising objective used for training.
- `val_var_bpb` is the compression-facing metric for this folder. It is a byte-normalized variational upper bound on NLL obtained by discretizing the same absorbing-mask process at evaluation time.

### Variational BPB

The variational BPB is not apples-to-apples comparable with the validation BPBs from the autoregressive models, which means this is a particularly special non-record submission. The variational metric was added because there is no perfect analogy to autoregressive models' losses in the diffusion regime:
- A masked diffusion model does not provide an exact autoregressive factorization of `p(x)` token-by-token, so the training loss is not directly an exact codelength.
- The exact codelength for the continuous-time process would require integrating over latent corruption trajectories, which is not tractable in this compact training script.
- To make compression comparison more apples-to-apples with AR baselines, eval instead reports a discrete absorbing-mask variational bound:
  - terminal KL term `KL(q(x_T | x_0) || p(x_T))`
  - plus a sum of reverse-process KL terms across `VAR_EVAL_STEPS`
- This is still an upper bound rather than an exact code length, but it is much more principled than simply converting the denoising loss into BPB units.
- This also allows us to measure the impact of discretization on the model by changing `VAR_EVAL_STEPS`

Command (track-relevant params):
```bash
RUN_ID=baseline_mdlm \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=8 \
records/track_non_record_16mb/2026-03-25-MaskedDiffusion/train_mdlm.py
```

Recommended knobs to sweep:
- `TRAIN_SEQ_LEN`: diffusion currently defaults to `256` rather than the AR baseline's `1024` because shorter windows improve throughput and increase the number of independent timestep samples per step.
- `SAMPLING_STEPS` / `SAMPLING_SCHEDULE`: test-time compute knobs for generation; they do not affect `val_var_bpb`.
- `VAR_EVAL_STEPS`: tighter but slower variational evaluation.
- `DROPOUT`: defaults to `0.0`, since masking noise already acts as a strong regularizer.

Current status:
- This folder is an in-progress diffusion adaptation rather than a final tuned submission.
- The script compiles and trains, but the compressed artifact currently exceeds the 16MB target at the default width.
- The largest size driver is the conditioning path (AdaLN modulation), even after reducing the internal conditioning width and restoring GQA math.

Files in this folder:
- `train_mdlm.py` - single-file masked diffusion training/eval script
- `command.txt` - launch commands used while iterating
- `dit.py`, `diffusion.py`, `noise_schedule.py`, `dataloader.py` - copied reference files from MDLM used for comparison during development

Metrics:
- Fill in `train.log` metrics here once a stable run completes.
- Suggested fields to record, matching other record READMEs:
  - best pre-quant `val_loss`
  - best pre-quant `val_var_bpb`
  - post-quant roundtrip `val_loss`
  - post-quant roundtrip `val_var_bpb`
  - step time / wallclock
  - compressed artifact size
