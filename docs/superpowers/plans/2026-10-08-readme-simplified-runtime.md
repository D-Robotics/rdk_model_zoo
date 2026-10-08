# Sample README Alignment Implementation Plan

> **For agentic workers:** Follow assigned README-only scopes and preserve benchmark and conversion source material.

**Goal:** Individually check all 51 first-party Samples, align README with simplified Runtime and retain benchmark/conversion information from available historical refs.

**Architecture:** Short model introductions; board support, preparation, conversion, APIs and benchmarks in their respective chapters. Review evidence lives outside README.

**Tech Stack:** Markdown, Git blob inventory, Python document checks.

**Spec:** User request dated 2026-10-08; docs/sample-standards/runtime-code.md and readme-contract.md.

## Global Constraints

Base ed4d9010. Preserve every benchmark value, unit, variant, board and measurement condition. Preserve conversion steps, commands, arguments, configurations, versions, links and images. No export/compiler or board execution in this fork. No migration, audit or editorial rationale in README. Keep bilingual anchors and real interfaces. ACT/Pi0 upstream untouched.

## Review Focus

Historical content absent from current README; ambiguous cross-board benchmark tables; stale runtime snippets; lost citations/requirements when shortening prose; complex or native samples falsely described as three Python files.

## Tasks

- [x] Index all locally available refs and unique README blobs.
- [x] Check/edit 22 classifiers, 17 other vision samples, 6 YOLO samples, 3 speech samples, HIMLoco and 2 LLM samples, individually.
- [x] Align shared entry documentation and documentation templates.
- [x] Compare benchmark rows and conversion fences/links/images, inspect historical additions and all removals.
- [x] Run document contract and relevant local link/snippet checks.
- [x] Review final diff and save per-sample completion evidence.
