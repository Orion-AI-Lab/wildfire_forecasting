#!/bin/bash
# Shedule execution of many runs
# Run from root folder with: bash bash/schedule.sh
export CUDA_VISIBLE_DEVICES=1

echo "Running experiment cnn_spatial_cls"
python run.py experiment=cnn_spatial_cls ++model.weight_decay=0.03 ++model.hidden_size=16 ++model.lr=0.0004

echo "Running experiment lstm_temporal_cls"
python run.py experiment=lstm_temporal_cls ++model.weight_decay=0.005 ++model.hidden_size=64 ++model.lr=0.001

echo "Running experiment clstm_spatiotemporal_cls"
python run.py experiment=clstm_spatiotemporal_cls ++model.weight_decay=0.03 ++model.hidden_size=16 ++model.lr=0.0002
