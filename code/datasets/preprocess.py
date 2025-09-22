from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def run_preprocess(project_root: Path) -> None:
    raw_path = project_root / "data" / "raw" / "titanic.csv"
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    dataset = pd.read_csv(raw_path)
    embarked = pd.get_dummies(dataset["Embarked"])
    gender = pd.get_dummies(dataset["Sex"])
    dataset = pd.concat([dataset, embarked, gender], axis=1)
    dataset = dataset.drop(["PassengerId", "Name", "Ticket", "Cabin", "Embarked", "Sex"], axis=1)
    for col in dataset.select_dtypes(include=["bool"]).columns:
        dataset[col] = dataset[col].astype(int)
    dataset = dataset.drop_duplicates()
    dataset["Age"] = dataset["Age"].fillna(dataset["Age"].median())

    X_train, X_test, y_train, y_test = train_test_split(
        dataset.drop("Survived", axis=1), dataset["Survived"], test_size=0.2, random_state=42
    )
    X_train.to_csv(processed_dir / "titanic_train.csv", index=False)
    X_test.to_csv(processed_dir / "titanic_test.csv", index=False)
    y_train.to_csv(processed_dir / "titanic_train_labels.csv", index=False)
    y_test.to_csv(processed_dir / "titanic_test_labels.csv", index=False)


