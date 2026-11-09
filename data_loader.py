import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


def load_books(books_dir: Path):
    if not books_dir.exists():
        raise FileNotFoundError(f"cant find books folder: {books_dir}")

    books = {}
    for f in sorted(books_dir.glob("*.txt")):
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
            # use the filename (no extension, lowercase) as the key
            books[f.stem.lower()] = text
            logger.info(f"loaded: {f.name} ({len(text):,} chars)")
        except Exception as e:
            logger.warning(f"couldnt read {f.name}: {e}")

    if not books:
        raise RuntimeError(f"no txt files in {books_dir}")

    logger.info(f"total books loaded: {len(books)}")
    return books


def load_train_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, dtype=str).fillna("")
    # drop rows with no content or label — they're useless
    df = df.dropna(subset=["content", "label"]).reset_index(drop=True)
    # normalise labels — the csv has at least one 'consistent    ' with trailing spaces
    df["label"] = df["label"].str.strip().str.lower()
    logger.info(f"train: {len(df)} rows | labels: {df['label'].value_counts().to_dict()}")
    return df


def load_test_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)

    # peek at the file to detect whether a header row is present
    raw = pd.read_csv(path, dtype=str, nrows=1)
    has_header = "content" in raw.columns

    if has_header:
        df = pd.read_csv(path, dtype=str).fillna("")
    else:
        # test1.csv has no header — same schema as train
        df = pd.read_csv(
            path,
            dtype=str,
            header=None,
            names=["id", "book_name", "char", "caption", "content"],
        ).fillna("")

    df = df[df["content"].str.strip() != ""].reset_index(drop=True)
    logger.info(f"test: {len(df)} rows | columns: {df.columns.tolist()}")
    return df
