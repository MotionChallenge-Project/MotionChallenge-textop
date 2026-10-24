# Inference validation

Author C++ hardware route: initial-heading reference alignment, all15 future position inputs zeroed by the author, and last action initialized from the author start pose. The exact hardware weight matches latest.onnx. Canonical benchmark physical reset is unchanged; no robot position is read.

[Script](../validate_inference.py) · [Author source pins](../native-sources.json) · [macOS report](native-parity.json) · [Linux report](native-parity-linux.json).

160 cases and4 independent resets, with the original checkpoint. Full benchmark results require separate complete-motion verification.
