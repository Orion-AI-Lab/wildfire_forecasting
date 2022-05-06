#!/bin/bash
# Shedule execution of many runs
# Run from root folder with: bash bash/schedule.sh
export CUDA_VISIBLE_DEVICES=1

echo "Running experiment optuna lstm_temporal_cls"
python run.py -m hparams_search=greecefire_optuna_grid experiment=lstm_temporal_cls >> optuna_runs_lstm.log &

echo "Running experiment optuna cnn_spatial_cls"
python run.py -m hparams_search=greecefire_optuna_grid experiment=cnn_spatial_cls >> optuna_runs_cnn.log

echo "Running experiment optuna clstm_spatiotemporal_cls"
python run.py -m hparams_search=greecefire_optuna_grid experiment=clstm_spatiotemporal_cls >> optuna_runs_cnn.log