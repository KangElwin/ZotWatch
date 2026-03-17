from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .faiss_store import FaissIndex
from .fuzzy_grey import fuzzy_grey_scores
from .models import CandidateWork, RankedWork
from .settings import Settings
from .vectorizer import TextVectorizer

logger = logging.getLogger(__name__)


@dataclass
class RankerArtifacts:
    index_path: Path
    profile_path: Path


class WorkRanker:
    def __init__(self, base_dir: Path | str, settings: Settings, vectorizer: TextVectorizer | None = None):
        self.base_dir = Path(base_dir)
        self.settings = settings
        self.vectorizer = vectorizer or TextVectorizer()
        self.artifacts = RankerArtifacts(
            index_path=self.base_dir / "data" / "faiss.index",
            profile_path=self.base_dir / "data" / "profile.json",
        )
        self.index = FaissIndex.load(self.artifacts.index_path)
        self.profile = self._load_profile()
        self.journal_metrics = self._load_journal_metrics()

    def _load_profile(self) -> dict:
        path = self.artifacts.profile_path
        if not path.exists():
            raise FileNotFoundError("Profile JSON not found; run profile build first.")
        return json.loads(path.read_text(encoding="utf-8"))

    def _load_journal_metrics(self) -> Dict[str, float]:
        path = self.base_dir / "data" / "journal_metrics.csv"
        metrics: Dict[str, float] = {}
        if not path.exists():
            logger.warning("Journal metrics file not found: %s", path)
            return metrics
        try:
            with path.open("r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    title = (row.get("title") or "").strip().lower()
                    sjr = row.get("sjr")
                    if not title or not sjr:
                        continue
                    try:
                        metrics[title] = float(sjr)
                    except ValueError:
                        continue
        except Exception as exc:
            logger.warning("Failed to load journal metrics: %s", exc)
            return {}
        logger.info("Loaded %d journal SJR entries", len(metrics))
        return metrics

    def rank(self, candidates: List[CandidateWork]) -> List[RankedWork]:
        if not candidates:
            return []

        texts = [c.content_for_embedding() for c in candidates]
        vectors = self.vectorizer.encode(texts)
        logger.info("Scoring %d candidate works", len(candidates))

        distances, _ = self.index.search(vectors, top_k=1)
        weights = self.settings.scoring.weights
        thresholds = self.settings.scoring.thresholds

        # Collect per-candidate raw sub-scores first (needed for fuzzy-grey path)
        raw: List[Tuple[float, float, float, float, float, float, float]] = []
        for candidate, distance in zip(candidates, distances):
            similarity = float(distance[0]) if distance.size else 0.0
            recency_score = _compute_recency(candidate.published, self.settings)
            citation_score, altmetric_score = _compute_metric(candidate)
            journal_quality, _ = _journal_quality_score(candidate.venue, self.journal_metrics)
            author_bonus = _bonus(candidate.authors, self.settings.scoring.whitelist_authors)
            venue_bonus = _bonus(
                [candidate.venue] if candidate.venue else [],
                self.settings.scoring.whitelist_venues,
            )
            raw.append((similarity, recency_score, citation_score, altmetric_score,
                        journal_quality, author_bonus, venue_bonus))

        # --- Scoring path --------------------------------------------------
        fg_cfg = self.settings.scoring.fuzzy_grey
        if fg_cfg.enabled and len(candidates) > 1:
            logger.info(
                "Using Fuzzy-Grey comprehensive evaluation (rho=%.2f, fuzzy_weight=%.2f)",
                fg_cfg.rho,
                fg_cfg.fuzzy_weight,
            )
            scores = self._fuzzy_grey_rank(raw, weights, fg_cfg)
        else:
            logger.info("Using linear weighted scoring")
            scores = self._linear_rank(raw, weights)

        # --- Build RankedWork objects ---------------------------------------
        ranked: List[RankedWork] = []
        for candidate, vector, distance, raw_scores, score in zip(
            candidates, vectors, distances, raw, scores
        ):
            similarity, recency_score, citation_score, altmetric_score, journal_quality, author_bonus, venue_bonus = raw_scores
            _, journal_sjr = _journal_quality_score(candidate.venue, self.journal_metrics)

            label = "ignore"
            if score >= thresholds.must_read:
                label = "must_read"
            elif score >= thresholds.consider:
                label = "consider"

            ranked.append(
                RankedWork(
                    **candidate.dict(),
                    score=score,
                    similarity=similarity,
                    recency_score=recency_score,
                    metric_score=citation_score,
                    author_bonus=author_bonus,
                    venue_bonus=venue_bonus,
                    journal_quality=journal_quality,
                    journal_sjr=journal_sjr,
                    label=label,
                )
            )
        ranked.sort(key=lambda w: w.score, reverse=True)
        return ranked

    # ------------------------------------------------------------------
    # Internal scoring helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _indicator_weights(weights) -> np.ndarray:
        """Return a numpy array of weights in indicator order."""
        return np.array([
            weights.similarity,
            weights.recency,
            weights.citations,
            weights.altmetric,
            getattr(weights, "journal_quality", 0.0),
            weights.author_bonus,
            weights.venue_bonus,
        ], dtype=float)

    def _linear_rank(self, raw, weights) -> List[float]:
        """Classic linear weighted sum scoring."""
        scores: List[float] = []
        for similarity, recency_score, citation_score, altmetric_score, journal_quality, author_bonus, venue_bonus in raw:
            score = (
                similarity * weights.similarity
                + recency_score * weights.recency
                + citation_score * weights.citations
                + altmetric_score * weights.altmetric
                + journal_quality * getattr(weights, "journal_quality", 0.0)
                + author_bonus * weights.author_bonus
                + venue_bonus * weights.venue_bonus
            )
            scores.append(score)
        return scores

    def _fuzzy_grey_rank(self, raw, weights, fg_cfg) -> List[float]:
        """Fuzzy-Grey comprehensive evaluation scoring."""
        indicator_matrix = np.array(raw, dtype=float)  # (n, 7)
        w = self._indicator_weights(weights)
        combined = fuzzy_grey_scores(
            indicator_matrix,
            w,
            rho=fg_cfg.rho,
            fuzzy_weight=fg_cfg.fuzzy_weight,
        )
        return combined.tolist()


def _bonus(values: List[str], whitelist: List[str]) -> float:
    whitelist_lower = {v.lower() for v in whitelist}
    for value in values:
        if value and value.lower() in whitelist_lower:
            return 1.0
    return 0.0


def _journal_quality_score(venue: Optional[str], metrics: Dict[str, float]) -> Tuple[float, Optional[float]]:
    if not venue:
        return 1.0, None
    key = venue.strip().lower()
    value = metrics.get(key)
    if value is None:
        return 1.0, None
    score = float(np.log1p(value))
    if score < 1.0:
        score = 1.0
    return score, float(value)


def _compute_recency(published: datetime | None, settings: Settings) -> float:
    if not published:
        return 0.0
    if published.tzinfo is None:
        published = published.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    delta_days = max((now - published).days, 0)
    decay = settings.scoring.decay_days
    if delta_days <= decay.get("fast", 30):
        return 1.0
    if delta_days <= decay.get("medium", 60):
        return 0.7
    if delta_days <= decay.get("slow", 180):
        return 0.4
    return 0.1


def _compute_metric(candidate: CandidateWork) -> Tuple[float, float]:
    citations = float(candidate.metrics.get("cited_by", candidate.metrics.get("is-referenced-by", 0.0)))
    altmetric = float(candidate.metrics.get("altmetric", 0.0))
    citation_score = float(np.log1p(citations)) if citations else 0.0
    altmetric_score = float(np.log1p(altmetric)) if altmetric else 0.0
    return citation_score, altmetric_score


__all__ = ["WorkRanker"]
