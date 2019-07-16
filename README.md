# Kali's tract notes

This branch holds my notes, not code.

These are personal notes, take everything here with a grain of salt. -- Kali

## Goal: Kaldi acoustic model runner

### Backlog: TODO

* establish an AM RTF bench
* declutter kaldi LSTM into scan + small pieces
* propagate Downsample Op upstream
* make Scan (and whatever else) pulse aware (-> stateful)
* optimize scan (codegen stage to generate plan)
* discover and apply concat/slice optimisation
* improve mat*vec product (i presume LSTM will generate lots of them)
* put test and benches in CI
* support inner networks profiling (dump done)

### Backlog: nice to have

* type the "incorporate" behaviour
* declutter ONNX black box recs and maybe TF LSTM to Scan + small pieces
* should Scan be rewritten in terms of Loop, Alloc, View, Assign ?
* optimize arithmetic noop (+0, *1 to generalize peepholes lstm op drops)

### Done

* 2019-07-11 We have a bunch of unit tests passing.
* 2019-07-15 Refactored core to be able to manipulate models with commong api. Made dump recursive.
* 2019-07-16 kaldi binary input

## Goal: refactor linalg kernels

* matmul and conv kernels in the shape of:
    * one linear stage (input + fma)
    * one intermediary stage (row bias or col bias, scale, clipt , transpose)
    * recombine and store

* first input is constant, and "packed"

* addressing for second input and output could be
    * row major or col major (one stride is 1)
    * two arbitrary stides
    * top value plus row offset (as the current direct conv)
