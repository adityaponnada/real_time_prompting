# Compute Raw Features Script

This script processes a sampled compliance matrix CSV file and computes a set of raw features for each prompt/observation. The output is a new CSV file with engineered features, ready for further analysis or modeling.

## Requirements
- Python 3.7+
- pandas

Install requirements (if needed):
```bash
pip install pandas numpy
```

## Usage

```bash
python compute_raw_features.py --input_csv <input_file.csv> --output_csv <output_file.csv>
```

- `--input_csv`: Path to the input sampled compliance matrix CSV file.
- `--output_csv`: Path where the processed CSV will be saved.

## Example

```bash
python compute_raw_features.py --input_csv sampled_compliance_matrix.csv --output_csv processed_compliance_matrix.csv
## Main Notebooks of Interest

The primary notebooks for analysis and modeling in this repository are:

- `general_model_between_user_with_SMOTE.ipynb`: General model for between-user analysis with SMOTE.
- `general_model_within_user_SMOTE.ipynb`: General model for within-user analysis with SMOTE.
- `personalized_user_models_SMOTE.ipynb`: Personalized user models with SMOTE.

Refer to these notebooks for the main workflows, modeling, and evaluation steps.
```

  - participant_id
  - prompt_time_converted
  - outcome
  - is_weekend
  - time_of_day
  - in_battery_saver_mode
  - location_category
  - screen_on
  - dist_from_home
  - is_phone_locked
  - wake_day_part
  - closeness_to_sleep_time
  - closeness_to_wake_time
  - mims_5min
  - days_in_study
  - completion_24h
  - completion_since_wake
  - completion_since_start

## Notes
- The script prints progress and summary information to the console.
- All feature engineering logic is contained within `compute_raw_features.py`.
