# """
# Model Evaluation
# """

# from sklearn.metrics import (
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     confusion_matrix,
#     classification_report,
# )


# def evaluate_model(model, X_test, y_test):

#     predictions = model.predict(X_test)

#     print("=" * 50)
#     print("Model Evaluation")
#     print("=" * 50)

#     print(f"Accuracy : {accuracy_score(y_test, predictions):.4f}")
#     print(f"Precision: {precision_score(y_test, predictions):.4f}")
#     print(f"Recall   : {recall_score(y_test, predictions):.4f}")
#     print(f"F1 Score : {f1_score(y_test, predictions):.4f}")

#     print("\nClassification Report\n")
#     print(classification_report(y_test, predictions))

#     return confusion_matrix(y_test, predictions)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    return {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1": f1_score(y_test, predictions),
    }