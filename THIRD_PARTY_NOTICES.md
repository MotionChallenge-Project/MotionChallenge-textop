# Third-party notices

This adapter follows [TextOp](https://github.com/TeleHuman/TextOp) at
`ef6555fb174c9b5c44945a62c7ffc77b5ddbbf22`. Its observation layout, joint order,
control constants and inference behavior come from the author's deployment code.
TextOp is copyright (c) 2025 TextOp Team and is distributed under the
[MIT license](licenses/TextOp-MIT.txt).

The quaternion arithmetic in `adapter.py` is adapted from
[Isaac Lab v2.1.0](https://github.com/isaac-sim/IsaacLab/tree/21f7136325136ca3f6ca4e0a8125edffe5c24f7e),
copyright (c) 2022–2025 The Isaac Lab Project Developers, under the
[BSD-3-Clause license](licenses/IsaacLab-BSD-3-Clause.txt).

`native-sources.json` identifies the exact author files used for independent
inference checks. Checkpoint files are downloaded from the author's published
artifact and are not included in this repository; their source and hashes are
recorded in `provenance.json`.
