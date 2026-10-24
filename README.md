# TextOp Tracker 75k

[Author project](https://github.com/TeleHuman/TextOp) · [Checkpoint](https://huggingface.co/datasets/Yochish/TextOp-Data/blob/bb9604ce36ad4c0a77b5cccde89ebbb7c3ca93c9/TextOpTracker/logs/rsl_rl/Pretrained/checkpoints/latest.onnx) · [Provenance](provenance.json)

## Install and run

```bash
git clone --recurse-submodules https://github.com/motion-challenge-project/motion-challenge-textop.git
cd motion-challenge-textop
uv sync
uv run evaluate.py
```

One full rollout per motion produces success, completion and all six MPBPE values
in one verified `result.json`. MuJoCo 3.17 CPU batch; `--jobs 8` sets rollout and
verification parallelism. Progress updates every five seconds.

Missing data and weights are downloaded and hash-checked. Cached files can be
selected with `--weights PATH --data PATH --offline`. Results and traces stay
outside Git.

This repository adapts model inference. The pinned root `motion-challenge`
submodule owns datasets, physics, metrics and verification. See `--help` for options.

Author inference checks: [script](validate_inference.py), [source pins](native-sources.json), [report](validation/native-parity.json).
