from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from src.models.classifier import SklearnClassifier
from src.utils.config import load_config
from src.utils.guardrails import validate_evaluation_metrics
from src.utils.store import AssignmentStore


@validate_evaluation_metrics
def main():
    store = AssignmentStore()
    config = load_config()

    df = store.get_processed("transformed_dataset.csv")
    df_train, df_test = train_test_split(df, test_size=config["test_size"])

    models = {
        'random_forest_default': RandomForestClassifier(**config["random_forest"]),
        'random_forest_tuned': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
        'logistic_regression': LogisticRegression(random_state=42),
        'Decision_tree': DecisionTreeClassifier(random_state=42)
    }

    best_model = None
    best_metrics = None
    best_score = -1
    all_metrics = {}

    for name, estimator in models.items():
        print(f"Training model: {name}")
        model = SklearnClassifier(estimator, config["features"], config["target"])
        model.train(df_train)
        metrics = model.evaluate(df_test)
        all_metrics[name] = metrics
        print(f"Metrics for {name}: {metrics}")

        if metrics['roc_auc'] > best_score:
            best_score = metrics['roc_auc']
            best_model = model
            best_metrics = metrics
            print(f"New best model: {name} with ROC AUC: {best_score}")

    print(f"\nBest model is: {type(best_model.clf).__name__} with metrics: {best_metrics}")
    store.put_model("saved_model.pkl", best_model)
    store.put_metrics("metrics.json", all_metrics)


if __name__ == "__main__":
    main()
