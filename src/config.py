"""Configuration settings for the project."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"

RAW_DATA = DATA_DIR / "raw" / "cs-training.csv"

PROCESSED_DATA = DATA_DIR / "processed"

MODEL_DIR = ROOT_DIR / "models"

IMAGE_DIR = ROOT_DIR / "images"
