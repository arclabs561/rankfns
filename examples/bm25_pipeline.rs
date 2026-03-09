//! End-to-end retrieval pipeline: `postings` for candidate generation, `rankfns` for scoring.
//!
//! Builds an in-memory postings index from a small document collection, retrieves candidates
//! via OR query, then scores each candidate under three models (BM25, TF-IDF, Dirichlet LM).
//! Prints ranked results for each, showing how different scoring functions reorder the same
//! candidate set.

use postings::PostingsIndex;
use rankfns::{
    bm25_idf_plus1, bm25_tf, idf_transform, lm_smoothed_p, tf_transform, IdfVariant,
    SmoothingMethod, TfVariant,
};

/// Corpus: each entry is (doc_id, title, word list).
/// Topics span rust programming, search engines, and machine learning.
fn corpus() -> Vec<(u32, &'static str, Vec<&'static str>)> {
    vec![
        (
            0,
            "Rust ownership",
            vec![
                "rust",
                "ownership",
                "borrow",
                "checker",
                "memory",
                "safety",
                "rust",
                "compiler",
            ],
        ),
        (
            1,
            "Search engine indexing",
            vec![
                "search",
                "engine",
                "index",
                "postings",
                "inverted",
                "query",
                "retrieval",
            ],
        ),
        (
            2,
            "BM25 ranking",
            vec![
                "search",
                "ranking",
                "bm25",
                "tf",
                "idf",
                "relevance",
                "scoring",
                "engine",
            ],
        ),
        (
            3,
            "Rust for search systems",
            vec![
                "rust",
                "search",
                "engine",
                "performance",
                "systems",
                "fast",
                "safe",
                "rust",
            ],
        ),
        (
            4,
            "ML fundamentals",
            vec![
                "machine",
                "learning",
                "gradient",
                "descent",
                "loss",
                "optimization",
                "model",
            ],
        ),
        (
            5,
            "Neural information retrieval",
            vec![
                "neural",
                "retrieval",
                "search",
                "embedding",
                "model",
                "ranking",
                "learning",
            ],
        ),
        (
            6,
            "Rust type system",
            vec![
                "rust", "type", "system", "generics", "traits", "compiler", "safety",
            ],
        ),
        (
            7,
            "Inverted index design",
            vec![
                "index",
                "postings",
                "compression",
                "search",
                "engine",
                "merge",
                "segments",
            ],
        ),
    ]
}

fn main() {
    // -- 1. Build the postings index --
    let docs = corpus();
    let titles: Vec<(u32, &str)> = docs.iter().map(|(id, title, _)| (*id, *title)).collect();
    let mut index = PostingsIndex::<String>::new();
    for (doc_id, _title, terms) in &docs {
        let owned: Vec<String> = terms.iter().map(|t| t.to_string()).collect();
        index
            .add_document(*doc_id, &owned)
            .expect("duplicate doc_id");
    }

    let n_docs = index.num_docs();
    let avg_dl = index.avg_doc_len();
    println!(
        "Indexed {} documents, avg length {:.1} terms",
        n_docs, avg_dl
    );

    // -- 2. Query: "rust search engine" --
    let query: Vec<String> = vec!["rust".into(), "search".into(), "engine".into()];
    let candidates = index.candidates(&query);
    println!(
        "\nQuery: {:?}\nCandidates (OR): {:?} ({} docs)\n",
        query,
        candidates,
        candidates.len()
    );

    // -- 3a. BM25 scoring --
    let k1 = 1.2_f32;
    let b = 0.75_f32;
    let mut bm25_scores: Vec<(u32, f32)> = candidates
        .iter()
        .map(|&doc_id| {
            let doc_len = index.document_len(doc_id) as f32;
            let score: f32 = query
                .iter()
                .map(|term| {
                    let df = index.df(term.as_str());
                    let tf = index.term_frequency(doc_id, term.as_str()) as f32;
                    bm25_idf_plus1(n_docs, df) * bm25_tf(tf, doc_len, avg_dl, k1, b)
                })
                .sum();
            (doc_id, score)
        })
        .collect();
    bm25_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("=== BM25 (k1={k1}, b={b}) ===");
    for (doc_id, score) in &bm25_scores {
        let title = title_for(*doc_id, &titles);
        println!("  doc {doc_id} [{title}]: {score:.4}");
    }

    // -- 3b. TF-IDF scoring (log-scaled TF, standard IDF) --
    let mut tfidf_scores: Vec<(u32, f32)> = candidates
        .iter()
        .map(|&doc_id| {
            let score: f32 = query
                .iter()
                .map(|term| {
                    let df = index.df(term.as_str());
                    let tf = index.term_frequency(doc_id, term.as_str());
                    tf_transform(tf, TfVariant::LogScaled)
                        * idf_transform(n_docs, df, IdfVariant::Standard)
                })
                .sum();
            (doc_id, score)
        })
        .collect();
    tfidf_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("\n=== TF-IDF (log-scaled TF, standard IDF) ===");
    for (doc_id, score) in &tfidf_scores {
        let title = title_for(*doc_id, &titles);
        println!("  doc {doc_id} [{title}]: {score:.4}");
    }

    // -- 3c. Language model scoring (Dirichlet, mu=1000) --
    // Corpus probability P(t|C) = total occurrences of t / total corpus length.
    let total_len: f32 = (0..n_docs).map(|id| index.document_len(id) as f32).sum();

    let mut lm_scores: Vec<(u32, f32)> = candidates
        .iter()
        .map(|&doc_id| {
            let doc_len = index.document_len(doc_id) as f32;
            // Query-likelihood: product of P(t|D) for each query term.
            // In log space to avoid underflow.
            let log_score: f32 = query
                .iter()
                .map(|term| {
                    let tf = index.term_frequency(doc_id, term.as_str()) as f32;
                    // Corpus frequency: sum of tf across all docs.
                    let cf: f32 = (0..n_docs)
                        .map(|id| index.term_frequency(id, term.as_str()) as f32)
                        .sum();
                    let p_corpus = if total_len > 0.0 { cf / total_len } else { 0.0 };
                    let p = lm_smoothed_p(
                        tf,
                        doc_len,
                        p_corpus,
                        SmoothingMethod::Dirichlet { mu: 1000.0 },
                    );
                    if p > 0.0 {
                        p.ln()
                    } else {
                        f32::NEG_INFINITY
                    }
                })
                .sum();
            (doc_id, log_score)
        })
        .collect();
    lm_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("\n=== Dirichlet LM (mu=1000, log query-likelihood) ===");
    for (doc_id, score) in &lm_scores {
        let title = title_for(*doc_id, &titles);
        println!("  doc {doc_id} [{title}]: {score:.4}");
    }

    // -- 4. Comparison summary --
    println!("\n=== Top-1 comparison ===");
    println!(
        "  BM25:     doc {} [{}]",
        bm25_scores[0].0,
        title_for(bm25_scores[0].0, &titles)
    );
    println!(
        "  TF-IDF:   doc {} [{}]",
        tfidf_scores[0].0,
        title_for(tfidf_scores[0].0, &titles)
    );
    println!(
        "  Dir. LM:  doc {} [{}]",
        lm_scores[0].0,
        title_for(lm_scores[0].0, &titles)
    );
}

fn title_for<'a>(doc_id: u32, titles: &'a [(u32, &'a str)]) -> &'a str {
    titles
        .iter()
        .find(|(id, _)| *id == doc_id)
        .map(|(_, t)| *t)
        .unwrap_or("?")
}
