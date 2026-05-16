#!/bin/bash
set -e
cd "$(dirname "$0")/.."

echo "========================================"
echo "  Building unit tests..."
echo "========================================"
make -f Makefile.simple test

echo ""
echo "========================================"
echo "  Running unit tests..."
echo "========================================"

FAILED=0
PASSED=0

for test_bin in unittest/test_matmul unittest/test_matadd unittest/test_relu \
                 unittest/test_softmax unittest/test_constant unittest/test_mlp; do
    echo ""
    echo "--- $test_bin ---"
    if ./"$test_bin"; then
        PASSED=$((PASSED + 1))
    else
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "========================================"
echo "  Results: $PASSED suites passed, $FAILED failed"
echo "========================================"
exit $FAILED
