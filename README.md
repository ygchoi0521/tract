# Kali's tract notes

This branch holds my notes, not code.

These are personal notes, take everything here with a grain of salt. -- Kali

## Goal: Kaldi acoustic model runner

### Done

* 2019-07-11 We have a bunch of unit tests passing.

### TODO

* binary input
* establish an AM RTF bench
* declutter kaldi LSTM into scan + small pieces
* propagate Downsample Op upstream
* make Scan (and whatever else) pulse aware (-> stateful)
* optimize scan (codegen stage to generate plan)
* discover and apply concat/slice optimisation
* improve mat*vec product (i presume LSTM will generate lots of them)

### Nice to have generalizations

* declutter ONNX black box recs and maybe TF LSTM to Scan + small pieces
* should Scan be rewritten in terms of Loop, Alloc, View, Assign ?
