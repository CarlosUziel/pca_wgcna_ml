import logging
import time
from collections import Counter
from typing import Callable, Dict, Iterable

import numpy as np
from Bio import Entrez
from rich.console import Console

console = Console()


def run_func_dict(kwargs: Dict, func: Callable):
    try:
        return func(**kwargs)
    except Exception as e:
        logging.error(e)
        console.print_exception(max_frames=20)
        return None


def trunc(values, decs=0):
    """Truncate float to a given number of decimal places."""
    return np.trunc(values * 10**decs) / (10**decs)


def search_db(query: str, email: str = "example@example.com", **kwargs) -> Dict:
    Entrez.email = email

    try:
        return Entrez.read(Entrez.esearch(term=query, **kwargs))
    except Exception as e:
        logging.error(e)
        time.sleep(1)
        return search_db(query, email, **kwargs)


def fetch_details_db(ids: Iterable[str], email: str = "example@example.com", **kwargs):
    Entrez.email = email

    try:
        return Entrez.read(Entrez.efetch(id=",".join(ids), **kwargs))
    except Exception as e:
        logging.error(e)
        time.sleep(1)
        return fetch_details_db(ids, email, **kwargs)


def get_papers_per_year(papers: Dict) -> Iterable[str]:
    papers_dates = []
    for paper in papers["PubmedArticle"]:
        try:
            papers_dates.append(
                paper["MedlineCitation"]["Article"]["Journal"]["JournalIssue"][
                    "PubDate"
                ]["Year"]
            )
        except KeyError:
            continue

    return Counter(papers_dates)
