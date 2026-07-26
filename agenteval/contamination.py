from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from agenteval.types import Task, TaskSuite

WORD_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(str(text).lower())


def ngrams(tokens: Sequence[str], n: int) -> set[str]:
    if len(tokens) < n:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def ngram_hashes(text: str, n: int = 8) -> set[str]:
    grams = ngrams(tokenize(text), n)
    return {hashlib.sha1(g.encode()).hexdigest()[:16] for g in grams}


@dataclass(frozen=True)
class ContaminationHit:
    task_id: str
    overlap: float
    matched_grams: int
    total_grams: int
    sample: str = ""


@dataclass
class ContaminationReport:
    n_gram: int
    threshold: float
    hits: list[ContaminationHit] = field(default_factory=list)
    checked: int = 0
    corpus_documents: int = 0

    @property
    def contaminated_ids(self) -> set[str]:
        return {h.task_id for h in self.hits}

    @property
    def contamination_rate(self) -> float:
        return len(self.hits) / self.checked if self.checked else 0.0

    @property
    def clean(self) -> bool:
        return not self.hits

    def summary(self, show: int = 10) -> str:
        lines = [
            f"  tasks checked       {self.checked}",
            f"  corpus documents    {self.corpus_documents}",
            f"  n-gram size         {self.n_gram}",
            f"  overlap threshold   {self.threshold:.0%}",
            f"  contaminated        {len(self.hits)} ({self.contamination_rate:.1%})",
        ]
        if self.hits:
            lines.append("")
            lines.append("  most contaminated tasks:")
            for h in sorted(self.hits, key=lambda x: -x.overlap)[:show]:
                lines.append(f"    {h.task_id:<20}{h.overlap:>7.1%}  ({h.matched_grams}/{h.total_grams} n-grams)")
                if h.sample:
                    lines.append(f"      matched: {h.sample[:70]!r}")
            lines.append("")
            lines.append("  Scores on these tasks may reflect memorization rather than capability.")
            lines.append("  Consider excluding them with suite.filter() before reporting.")
        else:
            lines.append("")
            lines.append("  No overlap above threshold detected.")
        return "\n".join(lines)


class CorpusIndex:
    def __init__(self, n_gram: int = 8):
        self._n = n_gram
        self._hashes: set[str] = set()
        self._documents = 0

    def add(self, text: str) -> None:
        self._hashes |= ngram_hashes(text, self._n)
        self._documents += 1

    def add_all(self, texts: Iterable[str]) -> None:
        for text in texts:
            self.add(text)

    def overlap(self, text: str) -> tuple[float, int, int]:
        grams = ngrams(tokenize(text), self._n)
        if not grams:
            return 0.0, 0, 0
        hashed = {(g, hashlib.sha1(g.encode()).hexdigest()[:16]) for g in grams}
        matched = [g for g, h in hashed if h in self._hashes]
        return len(matched) / len(grams), len(matched), len(grams)

    def matched_sample(self, text: str) -> str:
        grams = ngrams(tokenize(text), self._n)
        for g in grams:
            if hashlib.sha1(g.encode()).hexdigest()[:16] in self._hashes:
                return g
        return ""

    @property
    def size(self) -> int:
        return len(self._hashes)

    @property
    def documents(self) -> int:
        return self._documents


def detect_contamination(
    suite: TaskSuite,
    corpus: Iterable[str] | CorpusIndex,
    n_gram: int = 8,
    threshold: float = 0.5,
    include_expected: bool = True,
) -> ContaminationReport:
    index = corpus if isinstance(corpus, CorpusIndex) else CorpusIndex(n_gram)
    if not isinstance(corpus, CorpusIndex):
        index.add_all(corpus)

    report = ContaminationReport(
        n_gram=n_gram,
        threshold=threshold,
        checked=len(suite),
        corpus_documents=index.documents,
    )

    for task in suite.tasks:
        text = str(task.input)
        if include_expected and task.expected is not None:
            text = f"{text} {task.expected}"

        overlap, matched, total = index.overlap(text)
        if overlap >= threshold and total > 0:
            report.hits.append(ContaminationHit(
                task_id=task.id,
                overlap=overlap,
                matched_grams=matched,
                total_grams=total,
                sample=index.matched_sample(text),
            ))

    return report


def find_duplicates(suite: TaskSuite, n_gram: int = 5, threshold: float = 0.8) -> list[tuple[str, str, float]]:
    signatures = {t.id: ngrams(tokenize(str(t.input)), n_gram) for t in suite.tasks}
    ids = list(signatures)
    duplicates = []

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = signatures[ids[i]], signatures[ids[j]]
            if not a or not b:
                continue
            jaccard = len(a & b) / len(a | b)
            if jaccard >= threshold:
                duplicates.append((ids[i], ids[j], jaccard))

    return sorted(duplicates, key=lambda d: -d[2])


def clean_suite(suite: TaskSuite, report: ContaminationReport) -> TaskSuite:
    contaminated = report.contaminated_ids
    return TaskSuite(
        name=f"{suite.name}[decontaminated]",
        tasks=[t for t in suite.tasks if t.id not in contaminated],
        description=suite.description,
    )
