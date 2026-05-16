#!/bin/bash
set -e
cd "$(dirname "$0")/.."

echo "Cleaning build artifacts..."

rm -f toy_ml
rm -f unittest/test_matmul unittest/test_matadd unittest/test_relu
rm -f unittest/test_softmax unittest/test_constant unittest/test_mlp
rm -rf output

echo "Done. All build artifacts removed."
