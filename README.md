# TextOp Tracker 75k

[Author project](https://github.com/TeleHuman/TextOp) · [Checkpoint](https://huggingface.co/datasets/Yochish/TextOp-Data/blob/bb9604ce36ad4c0a77b5cccde89ebbb7c3ca93c9/TextOpTracker/logs/rsl_rl/Pretrained/checkpoints/latest.onnx) · [Provenance](provenance.json)

## Install and run

```bash
git clone --recurse-submodules https://github.com/MotionChallenge-Project/MotionChallenge-textop.git
cd MotionChallenge-textop
uv sync
uv run evaluate.py
```

One full rollout per motion produces success, completion and all six MPBPE values
in one verified `result.json`. Tuned inference defaults; CPU slots scale with
available cores. `--jobs N` overrides the slots. Progress updates every five seconds.

Missing data and weights are downloaded and hash-checked. Cached files can be
selected with `--weights PATH --data PATH --offline`. Results and traces stay
outside Git.

This repository adapts model inference. The pinned root `MotionChallenge-Core`
submodule owns datasets, physics, metrics and verification. See `--help` for options.

Author inference checks: [script](validate_inference.py), [source pins](native-sources.json), [report](validation/native-parity.json).

## Observations

Ideal proprioception, without robot world-position or contact feedback. [Checkpoint requirements](admission.json) · [Rules](https://github.com/MotionChallenge-Project/MotionChallenge-Core/blob/da0aec4023bbc3d29be67e1fce9fa9ab8a48c4b5/docs/OBSERVATIONS.md).

Author C++ hardware route: initial-heading reference alignment, all15 future position inputs zeroed by the author, and last action initialized from the author start pose. The exact hardware weight matches latest.onnx. Canonical benchmark physical reset is unchanged; no robot position is read.
