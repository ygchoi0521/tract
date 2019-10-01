# Kali's tract notes

This branch holds my notes, not code.

These are personal notes, take everything here with a grain of salt. -- Kali

## Epic: Kaldi acoustic model runner. (DONE. very satisafactory, not integrated)

### Backlog: TODO

* optim discover and apply concat/slice optimisation, then push them out of scan

### Backlog: nice to have

* establish an AM RTF bench: complex. hand-made measurement to 50% on a raspi 3b+
* type the "incorporate" behaviour
* declutter TF LSTM to Scan + small pieces
* should Scan be rewritten in terms of Loop, Alloc, View, Assign ? look at opset 11
* optimize arithmetic noop (+0, *1 to generalize peepholes lstm op drops)

### Done

* 2019-08 improve mat*vec product (i presume LSTM will generate lots of them)
* 2019-09 support inner networks profiling
* 2019-08 declutter kaldi LSTM into scan + small pieces
* 2019-08 propagate Downsample Op upstream
* 2019-08 make Scan (and whatever else) pulse aware (-> stateful)
* 2019-07-11 We have a bunch of unit tests passing.

* 2019-07-15 Refactored core to be able to manipulate models with commong api. Made dump recursive.
* 2019-07-16 kaldi binary input
* 2019-07-18 CI librispeech model
* 2019-07-19 plug regular optimisation into scan
* 2019-07-19 raspi3 bench, bactch mode 560ms for 2600 ms of signal :)

## Epic: Kaldi acoustic model runner, quantized

### 

## Epic: refactor linalg kernels. DONE

* matmul and conv kernels in the shape of:
    * one linear stage (input + fma)
    * one intermediary stage (row bias or col bias, scale, clipt , transpose)
    * recombine and store

* first input is constant, and "packed"

* addressing for second input and output could be
    * row major or col major (one stride is 1)
    * two arbitrary stides
    * top value plus row offset (as the current direct conv)

## Onnx

* convert LSTM and GRU to scan (DONE)
* address the dynamic tensors issue (really ?)

## Tensorflow

* Convert BlockLSTM to scan
* Convert arbitrary cells/nextiteration to scan

## Core improvements

* draft a uniform, typed, declutterred operator set (and type it ?) (ongoing)
* Consecutive reshaping must go
* noop reshaping must go
