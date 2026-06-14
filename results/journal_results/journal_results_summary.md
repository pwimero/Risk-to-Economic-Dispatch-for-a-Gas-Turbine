# Surrogate Journal Results Suite

- Created UTC: 2026-02-23T17:23:55.815189+00:00
- Data source: `/Users/primero/Library/CloudStorage/OneDrive-CityStGeorge's,UniversityofLondon/School/Year 4/INDIVIDUAL PROJECT/results/time_indexed_dataset.csv`
- Prepared features artifact: `/Users/primero/Library/CloudStorage/OneDrive-CityStGeorge's,UniversityofLondon/School/Year 4/INDIVIDUAL PROJECT/results/prepared_feature_dataset.joblib`
- Time split counts: train=25713, calib=5509, test=5511
- Total rows after lag pruning: 36709
- Unified feature count: 32

## Exported Tables

- `table_01_dataset_split_overview`: [`csv`](table_01_dataset_split_overview.csv), [`txt`](table_01_dataset_split_overview.txt), [`tex`](table_01_dataset_split_overview.tex)
- `table_02_dataset_variable_summary_overall`: [`csv`](table_02_dataset_variable_summary_overall.csv), [`txt`](table_02_dataset_variable_summary_overall.txt), [`tex`](table_02_dataset_variable_summary_overall.tex)
- `table_03_dataset_variable_summary_by_split`: [`csv`](table_03_dataset_variable_summary_by_split.csv), [`txt`](table_03_dataset_variable_summary_by_split.txt), [`tex`](table_03_dataset_variable_summary_by_split.tex)
- `table_04_feature_set_by_target`: [`csv`](table_04_feature_set_by_target.csv), [`txt`](table_04_feature_set_by_target.txt), [`tex`](table_04_feature_set_by_target.tex)
- `table_05_surrogate_hyperparameters`: [`csv`](table_05_surrogate_hyperparameters.csv), [`txt`](table_05_surrogate_hyperparameters.txt), [`tex`](table_05_surrogate_hyperparameters.tex)
- `table_06_trained_surrogate_registry`: [`csv`](table_06_trained_surrogate_registry.csv), [`txt`](table_06_trained_surrogate_registry.txt), [`tex`](table_06_trained_surrogate_registry.tex)
- `table_07_point_model_metrics_by_split`: [`csv`](table_07_point_model_metrics_by_split.csv), [`txt`](table_07_point_model_metrics_by_split.txt), [`tex`](table_07_point_model_metrics_by_split.tex)
- `table_08_quantile_model_metrics_by_split`: [`csv`](table_08_quantile_model_metrics_by_split.csv), [`txt`](table_08_quantile_model_metrics_by_split.txt), [`tex`](table_08_quantile_model_metrics_by_split.tex)
- `table_09_conformal_coverage_by_alpha`: [`csv`](table_09_conformal_coverage_by_alpha.csv), [`txt`](table_09_conformal_coverage_by_alpha.txt), [`tex`](table_09_conformal_coverage_by_alpha.tex)
- `table_10_crc_classifier_summary`: [`csv`](table_10_crc_classifier_summary.csv), [`txt`](table_10_crc_classifier_summary.txt), [`tex`](table_10_crc_classifier_summary.tex)
- `table_11_feature_importance_top20`: [`csv`](table_11_feature_importance_top20.csv), [`txt`](table_11_feature_importance_top20.txt), [`tex`](table_11_feature_importance_top20.tex)
- `table_12_target_correlation_matrix`: [`csv`](table_12_target_correlation_matrix.csv), [`txt`](table_12_target_correlation_matrix.txt), [`tex`](table_12_target_correlation_matrix.tex)

## Quick Look: Point-Model Test Metrics

```
target split  overfit     mae    rmse    medae       r2  mape_pct  smape_pct     bias  error_std  max_abs_error  abs_error_p50  abs_error_p90  abs_error_p95  pearson_r
   TIT  test     True 2.14752 2.94418  1.65406 0.976399  0.200226   0.199933 -1.62084    2.45786        19.9394        1.65406        4.85616         5.3849   0.991834
   NOX  test     True 3.90598 5.13014  3.31339 0.625158   7.16928    6.81639 -3.04674    4.12743        41.7812        3.31339        7.72415        9.44601   0.871552
    CO  test     True 0.45628 1.03551 0.275536 0.731171   16.8348    16.5831 0.203665    1.01529        34.4926       0.275536       0.904883        1.31694   0.861261
```
