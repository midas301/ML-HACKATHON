#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
python scripts/01_eda.py
python scripts/02_build_dataset.py
python scripts/03_train_models.py
python scripts/04_bayes_opt.py 30
python scripts/05_training_histories.py
python scripts/09_timing.py
python scripts/15_direction_experiment.py
python scripts/18_lightgbm_final.py
python scripts/06_hmm_eval.py
python scripts/07_maps.py
python scripts/08_compare_plots.py
python scripts/19_model_cards.py
python scripts/16_tuned_vs_untuned_table.py
python scripts/20_feature_speed_figures.py
python scripts/14_error_analysis.py > results/error_analysis.txt
python scripts/10_static_route_figure.py 00000056
python scripts/10_static_route_figure.py 00000077
python scripts/11_dataset_visual.py
python scripts/17_pipeline_figure.py
python scripts/22_report_data.py
node report/build_report.js
