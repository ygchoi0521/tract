#!/bin/sh

set -ex

export DEBIAN_FRONTEND=noninteractive

if [ `whoami` != "root" ]
then
    SUDO=sudo
fi

which rustup || curl https://sh.rustup.rs -sSf | sh -s -- -y
rustup update

. $HOME/.cargo/env

rustc --version

$SUDO apt-get -y update
$SUDO apt-get -y install binutils-aarch64-linux-gnu gcc-aarch64-linux-gnu libssl-dev pkg-config
rustup target add aarch64-unknown-linux-gnu
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc
(cd cli; cargo build --target aarch64-unknown-linux-gnu --release --no-default-features --features onnx)
