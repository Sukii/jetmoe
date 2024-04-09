#!/bin/bash
echo "Please wait for 10-15 minutes, as it pulls 40GB of model data files ..."
cd model/
./pull-model-files.sh
cd ..
cd chat-model/
./pull-model-files.sh
