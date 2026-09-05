# Notebook 4 — Feature Engineering & ML Dataset Preparation

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy import sparse

RANDOM_STATE = 42

# 1. Project root
PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_dataset.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "ml_ready"
MODEL_DIR = PROJECT_ROOT / "models" / "preprocessors"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("Project root:", PROJECT_ROOT)
print("Input:", DATA_PATH)

# 2. Load cleaned dataset
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype("string").str.strip()

print("Shape:", df.shape)
display(df.head())

# 3. Create targets from confirmed disease_flags mapping
# disease_flags = Diabetes, Hypertension, Heart Disease
def parse_flags(value):
    if pd.isna(value):
        return (np.nan, np.nan, np.nan)
    bits = str(value).strip().replace(" ", "")
    if bits not in {
        "0,0,0", "0,0,1", "0,1,0", "0,1,1",
        "1,0,0", "1,0,1", "1,1,0", "1,1,1"
    }:
        return (np.nan, np.nan, np.nan)
    return tuple(map(int, bits.split(",")))

flags = df["disease_flags"].apply(parse_flags)
df["diabetes_target"] = flags.apply(lambda x: x[0])
df["hypertension_target"] = flags.apply(lambda x: x[1])
df["heart_disease_target"] = flags.apply(lambda x: x[2])
df["obesity_target"] = df["bmi_level"].astype("string").str.strip()

target_cols = [
    "diabetes_target",
    "hypertension_target",
    "heart_disease_target",
    "obesity_target",
]

# 4. Validate targets
for col in target_cols:
    print("\n", col)
    print(df[col].value_counts(dropna=False).sort_index())

# Diabetes consistency check
if "diabetes" in df.columns:
    original = df["diabetes"].map({"Yes": 1, "No": 0})
    print("\nDiabetes mismatches:",
          (original != df["diabetes_target"]).sum())

# Remove rows missing any of the four targets
df_model = df.dropna(subset=target_cols).copy()
print("\nRows before:", len(df))
print("Rows after :", len(df_model))

# 5. Remove metadata, leakage, and redundant columns
GLOBAL_EXCLUDE = {
    "composite_key",
    "source_dataset",
    "disease_flags",
    "sublabel",
    "label",
    "diabetes",
    "hypertension",
    "heart_disease",
    "age_normalized",
    "bmi_level",
    "diabetes_target",
    "hypertension_target",
    "heart_disease_target",
    "obesity_target",
}

common_features = [
    c for c in df_model.columns if c not in GLOBAL_EXCLUDE
]

# Obesity: BMI is excluded because bmi_level is derived from BMI.
FEATURES = {
    "diabetes": common_features.copy(),
    "hypertension": common_features.copy(),
    "heart_disease": common_features.copy(),
    "obesity": [c for c in common_features if c != "bmi"],
}

for task, cols in FEATURES.items():
    print(f"\n{task.upper()}: {len(cols)} features")
    print(cols)
    print("Leakage overlap:", GLOBAL_EXCLUDE.intersection(cols))

# 6. Split function: 70/15/15 with stratification
def split_dataset(data, features, target):
    X = data[features].copy()
    y = data[target].copy()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

# 7. Preprocessor: fit only on training data
def make_preprocessor(X):
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    transformers = []
    if numeric:
        transformers.append(("num", num_pipe, numeric))
    if categorical:
        transformers.append(("cat", cat_pipe, categorical))

    return ColumnTransformer(transformers=transformers, remainder="drop"), numeric, categorical

# 8. Prepare and save four ML datasets
results = {}

for task, features in FEATURES.items():
    target = f"{task}_target"

    print("\n" + "=" * 80)
    print("TASK:", task.upper())

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        df_model, features, target
    )

    preprocessor, numeric, categorical = make_preprocessor(X_train)

    X_train_p = preprocessor.fit_transform(X_train)
    X_val_p = preprocessor.transform(X_val)
    X_test_p = preprocessor.transform(X_test)

    print("Raw:", X_train.shape, X_val.shape, X_test.shape)
    print("Processed:", X_train_p.shape, X_val_p.shape, X_test_p.shape)

    sparse.save_npz(PROCESSED_DIR / f"{task}_X_train.npz", X_train_p)
    sparse.save_npz(PROCESSED_DIR / f"{task}_X_val.npz", X_val_p)
    sparse.save_npz(PROCESSED_DIR / f"{task}_X_test.npz", X_test_p)

    y_train.to_csv(PROCESSED_DIR / f"{task}_y_train.csv", index=False)
    y_val.to_csv(PROCESSED_DIR / f"{task}_y_val.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / f"{task}_y_test.csv", index=False)

    # Save raw splits for inspection / explainability
    X_train.to_csv(PROCESSED_DIR / f"{task}_X_train_raw.csv", index=False)
    X_val.to_csv(PROCESSED_DIR / f"{task}_X_val_raw.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / f"{task}_X_test_raw.csv", index=False)

    joblib.dump(preprocessor, MODEL_DIR / f"{task}_preprocessor.joblib")

    metadata = {
        "task": task,
        "target": target,
        "raw_features": features,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "train_rows": len(X_train),
        "validation_rows": len(X_val),
        "test_rows": len(X_test),
        "processed_features": X_train_p.shape[1],
        "random_state": RANDOM_STATE,
    }

    with open(PROCESSED_DIR / f"{task}_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    results[task] = metadata

# 9. Verify class proportions
for task in FEATURES:
    target = f"{task}_target"
    print("\n" + "=" * 60)
    print(task.upper())

    for split in ["train", "val", "test"]:
        y = pd.read_csv(PROCESSED_DIR / f"{task}_y_{split}.csv")[target]
        print(f"\n{split}:")
        print(y.value_counts(normalize=True).sort_index())

# 10. Final summary
summary = pd.DataFrame([
    {
        "task": task,
        "target": meta["target"],
        "raw_features": len(meta["raw_features"]),
        "processed_features": meta["processed_features"],
        "train_rows": meta["train_rows"],
        "validation_rows": meta["validation_rows"],
        "test_rows": meta["test_rows"],
    }
    for task, meta in results.items()
])

display(summary)

print("\nML DATA PREPARATION COMPLETE.")
print("Processed data:", PROCESSED_DIR)
print("Preprocessors:", MODEL_DIR)

# Next: Notebook 5 — Baseline ML Models
