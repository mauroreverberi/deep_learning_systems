"""Create phrasebank.csv from the Financial PhraseBank v1.0 archive.

Usage: python prepare_dataset.py --download
       python prepare_dataset.py --source <FinancialPhraseBank-v1.0.zip>
"""

import argparse
import csv
import hashlib
import re
import sys
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

REVISION = "8d3fe0c36d5feec6b3cc5e455b0fcb4820fb9964"
SOURCE_URL = ("https://huggingface.co/datasets/takala/financial_phrasebank/resolve/"
              f"{REVISION}/data/FinancialPhraseBank-v1.0.zip")
SOURCE_SHA256 = "0e1a06c4900fdae46091d031068601e3773ba067c7cecb5b0da1dcba5ce989a6"
SEED = 42
LABELS = ("negative", "neutral", "positive")

# file name part and agreement level, from the loosest to the strictest file
AGREEMENT_FILES = [
    ("50Agree", "50-65%"),
    ("66Agree", "66-74%"),
    ("75Agree", "75-99%"),
    ("AllAgree", "100%"),
]
# threshold for grouping
SIMILARITY = 0.65
# shares of the template groups for test and validation, the rest is for training
TEST_SHARE = 0.15
VALIDATION_SHARE = 0.15


def sha256_of_file(path):
    """Return the SHA-256 hex digest of a file."""
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_sentences(archive, name):
    """Read one agreement file as a list of (sentence, label), split at the last @."""
    text = archive.read(f"FinancialPhraseBank-v1.0/Sentences_{name}.txt").decode("latin-1")
    rows = []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if not line.strip():
            continue
        sentence, label = line.rsplit("@", 1)
        if label.strip() not in LABELS:
            sys.exit(f"unknown label {label!r} in Sentences_{name}.txt")
        rows.append((sentence.strip(), label.strip()))
    return rows


def read_archive(source_path):
    """Read the four agreement files, keyed by their file name part."""
    with zipfile.ZipFile(source_path) as archive:
        return {name: read_sentences(archive, name) for name, _ in AGREEMENT_FILES}


def build_rows(files):
    """Join the files into one row per sentence with the level of the strictest file
    that has it. Sentences with two different labels are dropped, repeats kept once."""
    all_rows = files["50Agree"]
    labels_per_sentence = defaultdict(set)
    for sentence, label in all_rows:
        labels_per_sentence[sentence].add(label)
    conflicting = {s for s, labels in labels_per_sentence.items() if len(labels) > 1}

    level = {}
    for name, agreement in AGREEMENT_FILES:
        for pair in files[name]:
            level[pair] = agreement

    rows = []
    seen = set()
    dropped = Counter()
    for sentence, label in all_rows:
        if sentence in conflicting:
            dropped["conflicting"] += 1
            continue
        if sentence in seen:
            dropped["repeated"] += 1
            continue
        seen.add(sentence)
        rows.append({"sentence": sentence, "label": label, "agreement": level[(sentence, label)]})
    if len(rows) + dropped.total() != len(all_rows):
        sys.exit("The counts of kept and removed sentences do not add up.")
    print(f"{len(all_rows):,} sentences in the archive, {dropped['conflicting']} with conflicting "
          f"labels and {dropped['repeated']} repeated removed, {len(rows):,} kept")
    return rows


def find_links(sentences):
    """Return the index pairs of sentences that look like variants of the same template."""
    vectors = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit_transform(sentences)
    similar = cosine_similarity(vectors, dense_output=False)
    similar.data[similar.data < SIMILARITY] = 0  # keep only the pairs above the threshold
    similar.eliminate_zeros()
    pairs = [(int(i), int(j)) for i, j in zip(*similar.nonzero())]
    # sentences with the same letters are the same template with other numbers
    first_with_letters = {}
    for i, sentence in enumerate(sentences):
        letters = re.sub(r"[^a-z]", "", sentence.lower())
        pairs.append((i, first_with_letters.setdefault(letters, i)))
    return pairs


def assign_groups(rows):
    """Give every sentence the number of its template group, numbered in the order
    of the first sentence of each group."""
    neighbours = defaultdict(set)
    for i, j in find_links([row["sentence"] for row in rows]):
        neighbours[i].add(j)
        neighbours[j].add(i)
    group = 0
    for start in range(len(rows)):
        if "group" in rows[start]:
            continue
        # walk from this sentence to every sentence that is linked to it, directly or via others
        todo = [start]
        while todo:
            i = todo.pop()
            if "group" not in rows[i]:
                rows[i]["group"] = group
                todo.extend(neighbours[i])
        group += 1
    sizes = Counter(row["group"] for row in rows)
    groups = sum(1 for size in sizes.values() if size > 1)
    members = sum(size for size in sizes.values() if size > 1)
    print(f"{groups} template groups with {members} sentences")
    return rows


def assign_splits(rows):
    """Split the template groups about 70/15/15 into train, validation and test, so a
    group stays in one split."""
    stratum_of_group = {}
    for row in rows:
        stratum_of_group.setdefault(row["group"], f"{row['label']}|{row['agreement']}")
    groups = list(stratum_of_group)
    strata = [stratum_of_group[group] for group in groups]
    rest, test = train_test_split(groups, test_size=TEST_SHARE, stratify=strata, random_state=SEED)
    rest_strata = [stratum_of_group[group] for group in rest]
    train, validation = train_test_split(rest, test_size=VALIDATION_SHARE / (1 - TEST_SHARE),
                                         stratify=rest_strata, random_state=SEED)
    split_of_group = {}
    for split, members in (("train", train), ("validation", validation), ("test", test)):
        for group in members:
            split_of_group[group] = split
    for row in rows:
        row["split"] = split_of_group[row["group"]]
    return rows


def check_rows(rows):
    """Stop if a sentence appears twice or a group is spread over two splits."""
    if len({row["sentence"] for row in rows}) != len(rows):
        sys.exit("A sentence appears more than once.")
    splits_per_group = defaultdict(set)
    for row in rows:
        splits_per_group[row["group"]].add(row["split"])
    if any(len(splits) > 1 for splits in splits_per_group.values()):
        sys.exit("A template group is spread over more than one split.")


def print_splits(rows):
    """Print the size and the label shares of every split."""
    for split in ("train", "validation", "test"):
        labels = Counter(row["label"] for row in rows if row["split"] == split)
        total = sum(labels.values())
        shares = ", ".join(f"{label} {labels[label] / total:.1%}" for label in LABELS)
        print(f"{split}: {total:,} sentences, {shares}")


def write_output(rows, output_dir):
    """Write the rows as CSV in the original order and return the path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "phrasebank.csv"
    with data_path.open("w", newline="", encoding="utf-8") as stream:
        columns = ["sentence", "label", "agreement", "group", "split"]
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return data_path


def prepare(source_path, output_dir):
    """Run all steps and write phrasebank.csv into output_dir."""
    if sha256_of_file(source_path) != SOURCE_SHA256:
        sys.exit(f"{source_path} does not match the expected SHA-256. Download it with --download.")
    rows = build_rows(read_archive(source_path))
    rows = assign_splits(assign_groups(rows))
    check_rows(rows)
    print_splits(rows)
    data_path = write_output(rows, output_dir)
    print(f"wrote {data_path}, SHA-256 {sha256_of_file(data_path)}")


def download(target):
    """Download the archive of the fixed revision and check its checksum."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".partial")
    print(f"downloading {SOURCE_URL} ...", flush=True)
    urllib.request.urlretrieve(SOURCE_URL, partial)
    if sha256_of_file(partial) != SOURCE_SHA256:
        sys.exit("The downloaded file does not match the expected SHA-256.")
    partial.replace(target)


def main():
    """Parse the command line and run the preparation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/FinancialPhraseBank-v1.0.zip"),
        help="path of the archive (default: data/FinancialPhraseBank-v1.0.zip)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("."),
        help="folder for phrasebank.csv (default: current folder)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="download the archive first if --source does not exist",
    )
    args = parser.parse_args()
    if not args.source.exists():
        if args.download:
            download(args.source)
        else:
            sys.exit(f"{args.source} not found. Run with --download or pass --source.")
    prepare(args.source, args.output)


if __name__ == "__main__":
    main()
