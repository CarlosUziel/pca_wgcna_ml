import gzip
import logging
import shutil
from itertools import product
from pathlib import Path
from typing import Optional

import pandas as pd

from data.utils import parallelize_star


def unzip_gz(gz_file: Path, target_dir: Optional[Path]) -> None:
    """
    Unzips a .gz file to the specified target directory. If no target directory is provided,
    the file is unzipped in the same directory where it is located.

    Args:
        gz_file (Path): Path to the .gz file.
        target_dir (Optional[Path]): Directory where the file should be unzipped. If None,
            the file is unzipped in the same directory as gz_file.
    """
    target_dir = target_dir or gz_file.parent
    target_dir.mkdir(exist_ok=True, parents=True)
    dest_path = target_dir.joinpath(Path(gz_file).stem)

    with gzip.open(str(gz_file), "rb") as s_file, dest_path.open("wb") as d_file:
        shutil.copyfileobj(s_file, d_file)


def copy_file(file_path: Path, new_file_path: Path) -> None:
    """
    Copies a file from one path to another, ensuring that the parents of the new path exist.

    Args:
        file_path (Path): Path to the source file.
        new_file_path (Path): Path to the destination file.
    """
    new_file_path.parent.mkdir(exist_ok=True, parents=True)
    try:
        shutil.copy(file_path, new_file_path)
    except shutil.SameFileError as e:
        logging.warning(e)


def subset_star_counts(counts_file: Path, subset_col: int = 1) -> None:
    """
    Filters a STAR gene counts file by removing unnecessary rows and selecting the relevant column.

    Args:
        counts_file (Path): Path to the STAR gene counts file.
        subset_col (int): Index of the column to select. Defaults to 1.
    """
    df = pd.read_csv(counts_file, sep="\t", comment="#", index_col=0, header=None)

    if len(df.columns) == 1:
        logging.warning(
            "File only contains index plus one column, this star counts file"
            f" ({counts_file.name}) has already been filtered."
        )
        return

    df = df.loc[[idx for idx in df.index if "ENSG" in idx], subset_col].sort_index()
    df.to_csv(counts_file, sep="\t", header=False)


def clean_star_counts(
    star_path: Path, star_counts_path: Path, subset_col: int = 1
) -> None:
    """
    Copies and renames STAR counts files after mapping. It is assumed that gene counts files
    are inside directories named after the sample ID.

    Args:
        star_path (Path): Path to the directory containing STAR counts files.
        star_counts_path (Path): Path to the directory where cleaned counts files will be saved.
        subset_col (int): Index of the column to select. Defaults to 1.
    """
    star_counts_path.mkdir(exist_ok=True, parents=True)

    _ = parallelize_star(
        copy_file,
        [
            (counts_file, star_counts_path.joinpath(f"{counts_file.parent.stem}.tsv"))
            for counts_file in star_path.glob("*/ReadsPerGene.out.tab")
        ],
        method="fork",
    )

    _ = parallelize_star(
        subset_star_counts,
        list(product(star_counts_path.glob("*.tsv"), [subset_col])),
        method="fork",
    )


def rename_genes(counts_file: Path) -> None:
    """
    Renames genes in a counts file by removing the decimal part of the gene names.

    Args:
        counts_file (Path): Path to the tab-separated counts file.
    """
    df = pd.read_csv(counts_file, sep="\t", header=None, index_col=0)
    df.index = [idx.split(".")[0] for idx in df.index]
    df.to_csv(counts_file, sep="\t", header=False)


def intersect_raw_counts(counts_path: Path, pattern: str = "*.tsv") -> None:
    """
    Ensures that all raw count files in the specified directory contain only the intersection
    set of genes, i.e., genes that are present in all files.

    Args:
        counts_path (Path): Directory where the raw count files are located.
        pattern (str): Pattern to match the raw count files. Defaults to "*.tsv".
    """
    counts_files = list(counts_path.glob(pattern))
    assert len(counts_files) > 1, f"No files found under {counts_path}"

    counts_data = pd.DataFrame(
        {
            counts_file: {
                line.split("\t")[0]: line.split("\t")[1]
                for line in counts_file.read_text().splitlines()
            }
            for counts_file in counts_files
        }
    )

    counts_data.dropna(inplace=True)

    for c in counts_data.columns:
        counts_data[c].to_csv(c, sep="\t", header=False)
