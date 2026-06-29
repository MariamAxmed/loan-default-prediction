from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT_DIR / "images"

IMAGE_DIR.mkdir(exist_ok=True)


def plot_target_distribution(df):
    plt.figure(figsize=(6, 4))

    sns.countplot(data=df, x="SeriousDlqin2yrs")

    plt.title("Loan Default Distribution")

    plt.tight_layout()

    plt.savefig(IMAGE_DIR / "target_distribution.png", dpi=300)

    plt.show()


def plot_age_distribution(df):
    plt.figure(figsize=(8, 5))

    sns.histplot(df["age"], bins=30)

    plt.title("Age Distribution")

    plt.tight_layout()

    plt.savefig(IMAGE_DIR / "age_distribution.png", dpi=300)

    plt.show()


def plot_income_distribution(df):
    plt.figure(figsize=(8, 5))

    sns.histplot(df["MonthlyIncome"], bins=40)

    plt.title("Monthly Income Distribution")

    plt.tight_layout()

    plt.savefig(IMAGE_DIR / "income_distribution.png", dpi=300)

    plt.show()


def plot_correlation_heatmap(df):
    plt.figure(figsize=(12, 8))

    corr = df.corr(numeric_only=True)

    sns.heatmap(corr, cmap="coolwarm")

    plt.title("Correlation Matrix")

    plt.tight_layout()

    plt.savefig(IMAGE_DIR / "correlation_heatmap.png", dpi=300)

    plt.show()


def plot_boxplots(df):

    numeric_columns = [
        "age",
        "MonthlyIncome",
        "DebtRatio",
        "RevolvingUtilizationOfUnsecuredLines",
    ]

    for column in numeric_columns:

        plt.figure(figsize=(8, 4))

        sns.boxplot(x=df[column])

        plt.title(column)

        plt.tight_layout()

        plt.savefig(
            IMAGE_DIR / f"{column}_boxplot.png",
            dpi=300
        )

        plt.show()

from sklearn.metrics import ConfusionMatrixDisplay


def plot_confusion_matrix(model, X_test, y_test):

    plt.figure(figsize=(6, 6))

    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        cmap="Blues",
    )

    plt.title("Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        IMAGE_DIR / "confusion_matrix.png",
        dpi=300
    )

    plt.show()