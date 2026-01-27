//! `rankfns`: ranking math kernels for IR.
//!
//! This crate is intentionally **index-free**: it contains math transforms and scoring kernels
//! that can be used by multiple index structures (postings, positional, fielded, etc.).
//!
//! If you need an inverted index, use a structure crate (e.g. `postings`) and build rankers on top.

#![warn(missing_docs)]

/// Okapi/BM25-style IDF with a +1 inside the log to keep values non-negative.
///
/// \( \mathrm{idf} = \ln( ( (N - df + 0.5) / (df + 0.5) ) + 1 ) \)
pub fn bm25_idf_plus1(n_docs: f32, df: f32) -> f32 {
    if n_docs <= 0.0 || df <= 0.0 {
        return 0.0;
    }
    (((n_docs - df + 0.5) / (df + 0.5)) + 1.0).ln()
}

/// BM25 term-frequency normalization (the TF part).
pub fn bm25_tf(tf: f32, doc_len: f32, avg_doc_len: f32, k1: f32, b: f32) -> f32 {
    if tf <= 0.0 {
        return 0.0;
    }
    let avg = avg_doc_len.max(1e-9);
    let denom = tf + k1 * (1.0 - b + b * (doc_len / avg));
    (tf * (k1 + 1.0)) / denom.max(1e-9)
}

/// TF transform variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TfVariant {
    /// Linear TF: `tf`.
    Linear,
    /// Log-scaled TF: `1 + ln(tf)` for `tf > 0`.
    LogScaled,
}

/// IDF transform variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdfVariant {
    /// Standard IDF: `ln(N / df)`.
    Standard,
    /// Smoothed IDF: `ln(1 + (N - df + 0.5) / (df + 0.5))`.
    Smoothed,
}

/// TF transform.
pub fn tf_transform(tf: u32, variant: TfVariant) -> f32 {
    match variant {
        TfVariant::Linear => tf as f32,
        TfVariant::LogScaled => {
            if tf == 0 {
                0.0
            } else {
                1.0 + (tf as f32).ln()
            }
        }
    }
}

/// IDF transform.
pub fn idf_transform(n_docs: u32, df: u32, variant: IdfVariant) -> f32 {
    if n_docs == 0 || df == 0 {
        return 0.0;
    }
    let n = n_docs as f32;
    let d = df as f32;
    match variant {
        IdfVariant::Standard => (n / d).ln(),
        IdfVariant::Smoothed => (1.0 + (n - d + 0.5) / (d + 0.5)).ln(),
    }
}

/// Query-likelihood smoothing method (language-model retrieval).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SmoothingMethod {
    /// Jelinek–Mercer interpolation with `lambda` in `[0,1]`.
    JelinekMercer {
        /// Interpolation weight.
        lambda: f32,
    },
    /// Dirichlet smoothing with `mu >= 0`.
    Dirichlet {
        /// Prior strength.
        mu: f32,
    },
}

/// Compute a smoothed probability \(P(t|D)\) given:
/// - `tf`: term frequency in doc
/// - `doc_len`: document length
/// - `p_corpus`: corpus probability \(P(t|C)\)
pub fn lm_smoothed_p(tf: f32, doc_len: f32, p_corpus: f32, smoothing: SmoothingMethod) -> f32 {
    match smoothing {
        SmoothingMethod::JelinekMercer { lambda } => {
            let lam = lambda.clamp(0.0, 1.0);
            let p_doc = if doc_len > 0.0 { tf / doc_len } else { 0.0 };
            lam * p_doc + (1.0 - lam) * p_corpus
        }
        SmoothingMethod::Dirichlet { mu } => {
            let mu = mu.max(0.0);
            let denom = doc_len + mu;
            if denom > 0.0 {
                (tf + mu * p_corpus) / denom
            } else {
                0.0
            }
        }
    }
}

/// A generic trait for document retrieval.
pub trait Retriever {
    /// The type of query.
    type Query: ?Sized;
    /// The type of document identifiers.
    type DocId;

    /// Retrieve top-k documents for a given query.
    fn retrieve(
        &self,
        query: &Self::Query,
        k: usize,
    ) -> Result<Vec<(Self::DocId, f32)>, Box<dyn std::error::Error>>;
}
