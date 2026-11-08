# TACO algorithm source snapshot

The upstream files are frozen from `COCCL_TACO` revision
`ca717e05b6dd6d21fff4d2617d209c95be63f3ef` (`origin/taco-new-feature`).
The source working tree on `main` is unchanged; that branch contains the older
minmaxUint8 compressor and is not the TACO implementation. See `manifest.json`
for exact paths and SHA256 values.

These archived CUDA files are attribution and audit material, not the DCU build
input. The new implementation is `../../src/taco/codec.hip` and uses HIP shared
memory rather than CUDA warp32 intrinsics, software E4M3FN rather than hardware
FP8 conversion, and fixed H128 groups. Packet format is `[payload][qscale][ascale]`
with 136 bytes per 128 BF16 values. It retains TACO's scale/Hadamard arithmetic;
FP32 reduction order differs from CUDA and is checked against an independent
FP64 CPU oracle, not claimed bit-identical to the CUDA binary.

The surrounding fixed-source-order BF16 reduction is the DCU experiment's
protocol, not an assertion about all upstream COCCL collective algorithms.

Copyright and redistribution conditions are retained in `LICENSE.txt`.
