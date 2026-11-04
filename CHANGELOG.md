# Changelog

All notable changes to this project are documented here. Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.7] - 2026-11-04

### Fixed

- `IdfVariant::Standard` caps `df` at `n_docs`, as `IdfVariant::Smoothed`
  already did, so `df > n_docs` gives 0.0 instead of a negative IDF.

## [0.1.6] - 2026-11-01

### Changed

- `bm25_tf` and `lm_smoothed_p` return zero for non-finite inputs, compute in
  `f64`, and always return a finite, non-negative score for finite inputs.
  `lm_smoothed_p` caps `tf` at `doc_len`, so its result stays in `[0, 1]`.
- `IdfVariant::Smoothed` caps `df` at `n_docs`.
- Documented that `JelinekMercer::lambda` is the document-model weight (some
  Lucene-derived tools use it for the collection-model weight).
- The BM25 pipeline example uses `postings 0.6`.
- The published package ships only sources, tests, examples and docs, and
  `documentation` points at docs.rs.

## [0.1.5] - 2026-07-09

### Changed

- Updated the BM25 pipeline example to use `postings 0.4`.

## [0.1.4] - 2026-06-11

Maintenance release.

## [0.1.3] - 2026-04-06

### Added

- BM25 example, and a BM25 pipeline example with a postings index.

### Changed

- Tightened scope and hardened kernels.

[Unreleased]: https://github.com/arclabs561/rankfns/compare/v0.1.6...HEAD
[0.1.6]: https://github.com/arclabs561/rankfns/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/arclabs561/rankfns/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/arclabs561/rankfns/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/arclabs561/rankfns/releases/tag/v0.1.3
