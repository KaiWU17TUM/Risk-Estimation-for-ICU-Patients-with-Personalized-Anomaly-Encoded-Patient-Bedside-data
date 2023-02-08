# Risk-Estimation-for-ICU-Patients-with-Personalized-Anomaly-Encoded-Patient-Bedside-data


### Data preparation
Our experiements are conducted with [MIMIC-IV 1.0 database](https://physionet.org/content/mimiciv/1.0/).
To run the code, you need to
- download the database;
- follow this [tutorial](https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iv/buildmimic/postgres) to load MIMIC-IV into PostgreSQL database;
- run [notebooks/generate_all_mimic_dataset.ipynb](https://github.com/KaiWU17TUM/Risk-Estimation-for-ICU-Patients-with-Personalized-Anomaly-Encoded-Patient-Bedside-data/blob/main/notebooks/generate_all_mimic_dataset.ipynb) to read and save relevant data from the database;
- run [src/data_prep.py](https://github.com/KaiWU17TUM/Risk-Estimation-for-ICU-Patients-with-Personalized-Anomaly-Encoded-Patient-Bedside-data/blob/main/src/data_prep.py) to generate temporally-aligned 48H data samples in raw and anomaly-encoded format.


### Train models
- Run [train_cnn.sh](https://github.com/KaiWU17TUM/Risk-Estimation-for-ICU-Patients-with-Personalized-Anomaly-Encoded-Patient-Bedside-data/blob/main/train_cnn.sh) to train CNN models
- Run [train_xgb.sh](https://github.com/KaiWU17TUM/Risk-Estimation-for-ICU-Patients-with-Personalized-Anomaly-Encoded-Patient-Bedside-data/blob/main/train_xgb.sh) to train RandomForest or XGBoost models
