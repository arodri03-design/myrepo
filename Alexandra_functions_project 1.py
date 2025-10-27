"""
text_utils.py
Core text and data utilities for keyword work, URL checks, and CSV parsing.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from urllib.parse import urlparse
from io import StringIO
import csv
import re


class ValidationError(ValueError):
    """Raised when input validation fails."""


class DataParseError(ValueError):
    """Raised when data parsing fails."""


def validate_url_format(
    url: str,
    allowed_schemes: Iterable[str] = ("http", "https"),
) -> bool:
    """Quick format check for a URL without network calls. Simple tier function.

    Args:
        url: The URL string to validate.
        allowed_schemes: Iterable of acceptable schemes.

    Returns:
        True if the string looks like a valid URL, False otherwise.

    Raises:
        TypeError: If url is not a string.
        ValidationError: If allowed_schemes is empty or not iterable.

    Examples:
        >>> validate_url_format("https://example.com/page?q=1")
        True
        >>> validate_url_format("ftp://example.com")
        False
    """
    if not isinstance(url, str):
        raise TypeError("url must be a string")
    try:
        schemes = set(allowed_schemes)
    except TypeError as exc:
        raise ValidationError("allowed_schemes must be an iterable") from exc
    if not schemes:
        raise ValidationError("allowed_schemes must not be empty")

    parsed = urlparse(url.strip())
    if parsed.scheme not in schemes or not parsed.netloc:
        return False

    host = parsed.hostname or ""
    if host == "localhost":
        return True
    if "." not in host or host.startswith(".") or host.endswith("."):
        return False
    return True


def parse_csv_data(
    csv_text: str,
    *,
    has_header: bool = True,
    delimiter: str = ",",
    quotechar: str = '"',
    required_fields: Optional[Sequence[str]] = None,
    type_map: Optional[Dict[str, Callable[[str], object]]] = None,
    trim_whitespace: bool = True,
) -> Union[List[Dict[str, object]], List[List[str]]]:
    """Parse CSV content from a string with validation and optional typing.

    Medium tier function. Returns a list of dicts when the input has a header.
    Applies per column converters and reports row numbers on conversion errors.

    Args:
        csv_text: The CSV content as a single string.
        has_header: Treat the first row as a header.
        delimiter: Field delimiter character.
        quotechar: Quote character.
        required_fields: Column names that must be present when has_header is True.
        type_map: Mapping of column name to converter function.
        trim_whitespace: Strip whitespace on each field if True.

    Returns:
        List of dict rows when has_header is True, else list of lists.

    Raises:
        TypeError: If csv_text is not a string.
        ValidationError: If delimiter or quotechar are not single characters,
            or required fields are missing.
        DataParseError: If a value fails conversion in type_map.

    Examples:
        >>> txt = "name,qty,price\\nPen,5,1.5\\nPencil,3,0.5"
        >>> parse_csv_data(txt, type_map={"qty": int, "price": float})
        [{'name': 'Pen', 'qty': 5, 'price': 1.5}, {'name': 'Pencil', 'qty': 3, 'price': 0.5}]
    """
    if not isinstance(csv_text, str):
        raise TypeError("csv_text must be a string")
    if not isinstance(delimiter, str) or len(delimiter) != 1:
        raise ValidationError("delimiter must be a single character string")
    if not isinstance(quotechar, str) or len(quotechar) != 1:
        raise ValidationError("quotechar must be a single character string")

    reader = csv.reader(StringIO(csv_text), delimiter=delimiter, quotechar=quotechar)
    rows: List[List[str]] = []
    for r in reader:
        rows.append([c.strip() if trim_whitespace and isinstance(c, str) else c for c in r])

    if not rows:
        return [] if has_header else []

    if not has_header:
        return rows

    header = rows[0]
    data = rows[1:]
    if required_fields:
        missing = [f for f in required_fields if f not in header]
        if missing:
            raise ValidationError(f"Missing required fields: {', '.join(missing)}")

    index = {name: i for i, name in enumerate(header)}
    out: List[Dict[str, object]] = []
    for lineno, row in enumerate(data, start=2):
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        record: Dict[str, object] = {}
        for name, i in index.items():
            value = row[i]
            if type_map and name in type_map and value != "":
                try:
                    record[name] = type_map[name](value)
                except Exception as exc:
                    raise DataParseError(
                        f"Failed to convert column '{name}' on row {lineno}: {exc}"
                    ) from exc
            else:
                record[name] = value
        out.append(record)
    return out


def extract_keywords(
    text: str,
    *,
    min_length: int = 3,
    stopwords: Optional[Iterable[str]] = None,
    top_n: Optional[int] = None,
    keep_case: bool = False,
    return_counts: bool = False,
    include_bigrams: bool = False,
) -> Union[List[str], List[Tuple[str, int]]]:
    """Extract likely keywords from free text using frequency and filters.

    Complex tier function. It tokenizes the text, normalizes case when desired,
    removes short tokens and user stopwords, counts frequencies, and returns
    the most frequent items. It can also build bigrams and return counts.

    Args:
        text: The input text to analyze.
        min_length: Minimum token length to keep. Must be at least 1.
        stopwords: Optional words to exclude. Respected with case rules.
        top_n: If set, limit the returned list to the top N items.
        keep_case: When False the text is lowercased.
        return_counts: When True return (term, count) tuples.
        include_bigrams: When True also count two word phrases.

    Returns:
        List of strings ordered by frequency, or list of (string, count) pairs
        when return_counts is True.

    Raises:
        TypeError: If text is not a string.
        ValidationError: If min_length is less than 1 or top_n is not positive.

    Examples:
        >>> extract_keywords("Banana bread is great. Banana pie is also great.")
        ['banana', 'great', 'is', 'also', 'bread', 'pie']
        >>> extract_keywords("a a a bb bb ccc", min_length=2, return_counts=True)
        [('bb', 2), ('ccc', 1)]
        >>> extract_keywords("data science rocks", include_bigrams=True, top_n=3)
        ['data', 'science', 'data science']
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(min_length, int) or min_length < 1:
        raise ValidationError("min_length must be an integer greater than or equal to 1")
    if top_n is not None and (not isinstance(top_n, int) or top_n <= 0):
        raise ValidationError("top_n must be a positive integer when provided")

    # Normalize case
    processed = text if keep_case else text.lower()

    # Basic tokenization. Keep letters and internal apostrophes.
    tokens = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", processed)

    # Prepare stopword set with the same case rule
    if stopwords is None:
        stop_set: set = set()
    else:
        stop_set = set(stopwords if keep_case else (w.lower() for w in stopwords))

    # Filter tokens by length and stopwords
    words: List[str] = [t for t in tokens if len(t) >= min_length and t not in stop_set]

    # Optionally build bigrams
    terms: List[str] = list(words)
    if include_bigrams and len(words) >= 2:
        bigrams = [f"{a} {b}" for a, b in zip(words, words[1:])]
        terms.extend(bigrams)

    # Count and sort by frequency then alphabetically for stable order
    counts = Counter(terms)
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))

    if top_n is not None:
        items = items[:top_n]

    if return_counts:
        return items
    return [k for k, _ in items]
