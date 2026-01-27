# rankfns

Small ranking-math kernels for IR (no indexing).

## Scope

- BM25/TF-IDF/LM-style transforms and helpers
- Small trait (`Retriever`) for “top-k by query” adapters

If you want an inverted index, pair this with a structure crate like `postings` or a scorer crate like `lexir`.

