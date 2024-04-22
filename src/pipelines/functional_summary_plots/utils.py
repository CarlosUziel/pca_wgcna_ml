from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud


def generate_wordcloud(docs: Iterable[str], save_path: Path, results_name: str) -> None:
    """Generate a world-cloud from a set of documents

    Args:
        docs: Iterable of documents, represented as strings.
        save_path: Path to save worldcloud.
        results_name: Name of the file.
    """
    # 1. Join all term descriptions
    lemmatizer = WordNetLemmatizer()
    wc = Counter(
        [
            lemmatizer.lemmatize(lemmatizer.lemmatize(token), pos="v")
            for token in word_tokenize(" ".join(docs))
            if token not in stopwords.words("english")
        ]
    )
    max_count = max(wc.values())
    wc = {word: count / max_count for word, count in wc.items()}

    # 2. Get word cloud
    wordcloud = WordCloud(
        width=2500,
        height=2500,
        max_words=100,
        background_color="white",
        color_func=lambda word, **kwargs: f"hsl({200 + 160 * wc[word]}, 100%, 50%)",
    ).generate_from_frequencies(wc)

    # 3. Save word cloud
    wordcloud.to_file(save_path.joinpath(f"{results_name}_wordcloud.pdf"))

    wordcloud_svg = wordcloud.to_svg(embed_font=True)
    save_path.joinpath(f"{results_name}_wordcloud.svg").open("w+").write(wordcloud_svg)


def functional_summary_plots(
    functional_result_file: Path,
    save_path: Path,
) -> None:
    """
    Given a functional results file, generate a word cloud summarizing it.

    Args:
        functional_result_file: A file containing functional results.
        save_path: Directory to store the word cloud image file.
    """
    # 0. Setup
    assert functional_result_file.exists(), "Functional result file does not exist."

    # 1. Load functional result
    func_result = pd.read_csv(Path(functional_result_file), index_col=0)

    # 2. Word cloud
    generate_wordcloud(
        func_result["Description"].tolist(), save_path, functional_result_file.stem
    )
