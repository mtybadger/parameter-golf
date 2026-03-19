This record contains three main new ideas, as well as some tweaks to the baseline, particularly vocab size. I had several ideas I wanted to try today, and these are the ones that worked - I want to chase further on quantization in the coming days.

Changes in this model:
- Vocab size 1024 -> 8192
- New "sp8192" tokenizer trained using `./data/download_hf_docs_and_tokenize.py   --output-root ./data   --tokenizer-config ./data/tokenizer_specs.json --max-train-tokens 8000000000 --tokenizer-train-docs 100000`, for a 50/50 val/train split
- NorMuon implementation from [the original paper](https://github.com/zichongli5/NorMuon), popularized by `modded-nanogpt`, replacing Muon
- Selective Quantization: the weights are quantized to int6, while the embeddings are kept at int8. Not sure if this is optimal and have seen plenty of weird behaviour from this, but I think it's in the right direction; I think quantization will be really key to this challenge and I want to dig into it more. From now on there will be a lot of trading off flexibility in various parameters, and I think selectively quantizing some more than others will allow us to do that more fine-grained!

Configuration:
- All hyperparams as in default NaiveBaseline except VOCAB_SIZE

Command:
```bash
NCCL_IB_DISABLE=1 \
RUN_ID=hf_verify_sp1024_8gpu \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=200 \
torchrun --standalone --nproc_per_node=8 ./records/track_10min_16mb/2026-03-19_NorMuon+SelectiveQuant/train_gpt.py
```

Command:
```bash
NCCL_IB_DISABLE=1 \
RUN_ID=verify_sp8192_w6e8_8gpu \
DATA_PATH=./data/datasets/fineweb10B_sp8192 \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model \
VOCAB_SIZE=8192 \
MAX_WALLCLOCK_SECONDS=600 \
WEIGHT_QUANTIZATION_BITS=6 \
EMBED_QUANTIZATION_BITS=8 \
WARMDOWN_ITERS=3000 \
TRAIN_SEQ_LEN=2048 \
NUM_LAYERS=8 \
torchrun --standalone --nproc_per_node=8 ./records/track_10min_16mb/2026-03-19_VocabSize_NorMuon_SelectiveQuant/train_gpt.py
```

Key metrics (from `train.log`):
- Timed training stopped at `13780/20000` steps due to the wallclock cap.
- Pre-quant eval at stop: `val_loss:2.0606`, `val_bpb:1.2172`
- Post-quant roundtrip eval: `val_loss:2.0727`, `val_bpb:1.2244`
- Exact printed metric: `final_int8_zlib_roundtrip_exact val_bpb:1.22436570`
- Train time: `600038ms` (`step_avg:43.54ms`)
- Peak memory: `10184 MiB allocated`, `10200 MiB reserved`
- Serialized model int8+zlib: `15815847 bytes`
- Code size: `47642 bytes`
- Total submission size int8+zlib: `15863489 bytes`

Training volume:
- Global batch: `524288` tokens/step
- Total train tokens seen: `7224688640`

Included files:
- `train_gpt.py` (code snapshot used for the run)
- `train.log` (exact remote training log)
- `submission.json` (leaderboard metadata)
