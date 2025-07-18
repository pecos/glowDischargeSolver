#!/bin/bash

./testRunBase.sh || echo "testRunBase.sh failed"
./testRunCN.sh || echo "testRunCN.sh failed"
./testInterpTrans.sh || echo "testInterpTrans.sh failed"
./testRunBackground.sh || echo "testRunBackground.sh failed"
./test6Species.sh || echo "test6Species.sh failed"
./testCRModel.sh || echo "testCRModel.sh failed"
