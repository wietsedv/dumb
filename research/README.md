# Research

Scripts in this directory were used for the paper. These scripts are not necessary for using DUMB or reproducing results. All data is available in `/hf_datasets` and training scripts are in `/trainers`.

## Data
- `datasets_list.py` show dataset sizes

## Training
- `missing_train.py` show missing training runs
- `missing_eval.py` show missing test runs

## Results tables
- `list.py` show all results
- `best.py` show best results in markdown or latex


## Paper tables
- Table 1: `python research/datasets_list.py`
- Table 2: `python research/eval.py`
- Table 3: `python research/table_correlate.py`
- Table 4: `python research/table_stats.py`


## Adding a model

1. Add identfiers to `research/constants.py` and `trainers/run.sh`
2. Grid search: `python research/missing_train.py -m {model_id}`
3. Eval seeds: `python research/missing_eval.py -m {model_id}`
4. Keep track of runs: `python research/list.py -m {model_id}`
5. Show results: `python research/best.py -m {model_id} | pandoc -t plain`
6. Export predictions: `python research/export_predictions.py -m {model_id}`
6. Export results: `python research/eval.py`
