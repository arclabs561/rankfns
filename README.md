# rankfns

Small ranking-math kernels for IR (no indexing).

## Scope

- **BM25 / TF-IDF**: Standard scoring functions (Robertson-Walker, etc.).
- **Normalization**: Document length normalization helpers.
- **Language-model smoothing**: query-likelihood components (Jelinek–Mercer, Dirichlet).

If you want an inverted index, pair this with `postings` (storage) or `lexir` (scoring pipeline).

## Example (BM25 pieces)

```rust
use rankfns::{bm25_idf_plus1, bm25_tf};

let n_docs = 1_000u32;
let df = 10u32;
let tf = 3.0f32;
let doc_len = 120.0f32;
let avg_doc_len = 100.0f32;
let k1 = 1.2f32;
let b = 0.75f32;

let idf = bm25_idf_plus1(n_docs, df);
let tf_norm = bm25_tf(tf, doc_len, avg_doc_len, k1, b);
let score = idf * tf_norm;

assert!(score.is_finite());
assert!(score >= 0.0);
```