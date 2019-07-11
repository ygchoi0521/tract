# Kali's tract notes

This branch holds my notes, not code.

These are personal notes, take everything here with a grain of salt. -- Kali

## Goal: Kaldi acoustic model runner

### Done

* 2019-07-11 We have a bunch of unit tests passing.

### TODO

* declutter kaldi LSTM into scan + small pieces
* propagate downsample upstream
* make Scan pulse aware
* discover concat/slice optimisation, (kaldi ops never use multiple inputs)

### Generalizations

* declutter ONNX black box recs and maybe TF LSTM to Scan + small pieces
* should Scan be rewritten in terms of Loop, Alloc, View, Assign ?
