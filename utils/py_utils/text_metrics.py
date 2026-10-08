# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Deterministic character edit counts; no silent language normalization."""


def edit_counts(reference, hypothesis):
    """Levenshtein counts with diagonal, deletion, insertion tie preference.

    Inputs are Unicode strings; each Python code point is one character.
    Uses one dynamic-programming row, O(len(hypothesis)) memory.
    """
    if not isinstance(reference, str) or not isinstance(hypothesis, str):
        raise ValueError("Reference and hypothesis must be strings")
    # Tuples carry (distance, substitutions, deletions, insertions).
    previous = [(i, 0, 0, i) for i in range(len(hypothesis) + 1)]
    for i, left in enumerate(reference, 1):
        current = [(i, 0, i, 0)]
        for j, right in enumerate(hypothesis, 1):
            cost, s, d, n = previous[j - 1]
            different = int(left != right)
            diagonal = (cost + different, s + different, d, n)
            cost, s, d, n = previous[j]
            deletion = (cost + 1, s, d + 1, n)
            cost, s, d, n = current[j - 1]
            insertion = (cost + 1, s, d, n + 1)
            current.append(min((diagonal, deletion, insertion), key=lambda row: row[0]))
        previous = current
    distance, s, d, i = previous[-1]
    return dict(substitutions=s, deletions=d, insertions=i, distance=distance)


def score_transcripts(records):
    """Micro-averaged CER, retaining per-utterance errors and empty references."""
    if not isinstance(records, list) or not records:
        raise ValueError("Expected a nonempty transcript list")
    seen = set()
    items = []
    totals = dict(substitutions=0, deletions=0, insertions=0, distance=0)
    reference_characters = 0
    exact_matches = 0
    for record in records:
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("id"), str)
            or not record["id"]
            or record["id"] in seen
        ):
            raise ValueError("Transcript IDs must be unique nonempty strings")
        seen.add(record["id"])
        reference = record.get("reference")
        hypothesis = record.get("hypothesis")
        errors = edit_counts(reference, hypothesis)
        for key in totals:
            totals[key] += errors[key]
        reference_characters += len(reference)
        exact_matches += int(reference == hypothesis)
        items.append(
            dict(
                id=record["id"],
                reference=reference,
                hypothesis=hypothesis,
                reference_characters=len(reference),
                **errors
            )
        )
    return dict(
        count=len(records),
        reference_characters=reference_characters,
        exact_matches=exact_matches,
        cer=totals["distance"] / reference_characters if reference_characters else None,
        errors=totals,
        utterances=items,
        normalization="none; Unicode code points; whitespace/case/punctuation retained",
    )
