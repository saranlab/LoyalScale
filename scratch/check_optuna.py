import optuna
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
db_path = os.path.join(BASE_DIR, "processed_data", "optuna_study_telecom.db")

if not os.path.exists(db_path):
    print(f"Database not found at: {db_path}")
    sys.exit(1)

storage_url = f"sqlite:///{db_path}"
try:
    study = optuna.load_study(study_name="optuna_study_telecom", storage=storage_url)
    trials = study.trials
    print(f"Total trials found: {len(trials)}")
    for i, t in enumerate(trials[-5:]):
        print(f"Trial {t.number}: state={t.state}, value={t.value}, params={t.params}")
except Exception as e:
    print(f"Error loading study: {e}")
