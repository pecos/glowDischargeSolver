#!/bin/bash

./testRunBase.sh || echo "testRunBase.sh failed"
./testRunCN.sh
./testInterpTrans.sh
./testRunBackground.sh
./test6Species.sh
