//! `rankfns`: ranking math kernels for IR.
//!
//! This crate is intentionally **index-free**: it contains math transforms and scoring kernels
//! that can be used by multiple index structures (postings, positional, fielded, etc.).
//!
//! If you need an inverted index, use a structure crate (e.g. `postings`) and build rankers on top.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

/// Okapi/BM25-style IDF with a +1 inside the log to keep values non-negative.
///
/// \( \mathrm{idf} = \ln( ( (N - df + 0.5) / (df + 0.5) ) + 1 ) \)
///
/// Robustness notes:
/// - `df == 0` returns 0.0 (no evidence).
/// - `df > n_docs` is clamped to `n_docs` to avoid `NaN` from an invalid log argument.
pub fn bm25_idf_plus1(n_docs: u32, df: u32) -> f32 {
    if n_docs == 0 || df == 0 {
        return 0.0;
    }
    let n = n_docs as f32;
    let d = (df.min(n_docs)) as f32;
    (((n - d + 0.5) / (d + 0.5)) + 1.0).ln()
}

/// BM25 term-frequency normalization (the TF part).
///
/// Robustness notes:
/// - `tf <= 0` returns 0.0.
/// - Any non-finite input returns 0.0.
/// - Negative document length is treated as zero; negative `k1` is treated as
///   zero; `b` is clamped to `[0, 1]`; and average length is clamped away from
///   zero.
/// - Finite inputs always produce a finite, non-negative result.
pub fn bm25_tf(tf: f32, doc_len: f32, avg_doc_len: f32, k1: f32, b: f32) -> f32 {
    if !tf.is_finite()
        || !doc_len.is_finite()
        || !avg_doc_len.is_finite()
        || !k1.is_finite()
        || !b.is_finite()
        || tf <= 0.0
    {
        return 0.0;
    }
    let tf = f64::from(tf);
    let doc_len = f64::from(doc_len.max(0.0));
    let avg = f64::from(avg_doc_len.max(1e-9));
    let k1 = f64::from(k1.max(0.0));
    let b = f64::from(b.clamp(0.0, 1.0));
    let length_norm = 1.0 - b + b * doc_len / avg;
    let score = (k1 + 1.0) / (1.0 + k1 * length_norm / tf);
    score.min(f64::from(f32::MAX)) as f32
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
    match variant {
        IdfVariant::Standard => (n / df as f32).ln(),
        IdfVariant::Smoothed => {
            let d = df.min(n_docs) as f32;
            (1.0 + (n - d + 0.5) / (d + 0.5)).ln()
        }
    }
}

/// Query-likelihood smoothing method (language-model retrieval).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SmoothingMethod {
    /// Jelinek–Mercer interpolation with a document-model weight in `[0, 1]`.
    ///
    /// This crate computes `lambda * P(t|D) + (1 - lambda) * P(t|C)`.
    /// Lucene-derived tools commonly use `lambda` for the collection-model
    /// weight instead; convert such a value with `1 - lambda`.
    JelinekMercer {
        /// Weight assigned to the document model.
        lambda: f32,
    },
    /// Dirichlet smoothing with `mu >= 0`.
    Dirichlet {
        /// Prior strength.
        mu: f32,
    },
}

impl Default for SmoothingMethod {
    fn default() -> Self {
        // Conventional “large-ish” value used in many IR baselines.
        Self::Dirichlet { mu: 1000.0 }
    }
}

/// Compute a smoothed probability \(P(t|D)\) given:
/// - `tf`: term frequency in doc
/// - `doc_len`: document length
/// - `p_corpus`: corpus probability \(P(t|C)\)
///
/// Non-finite inputs and negative `tf` or `doc_len` return zero. For finite
/// inputs, `p_corpus` and interpolation weights are clamped to their valid
/// ranges, `mu` is clamped to zero, and `tf` is capped at `doc_len`. The result
/// is therefore a finite probability in `[0, 1]`.
pub fn lm_smoothed_p(tf: f32, doc_len: f32, p_corpus: f32, smoothing: SmoothingMethod) -> f32 {
    if !tf.is_finite() || !doc_len.is_finite() || !p_corpus.is_finite() || tf < 0.0 || doc_len < 0.0
    {
        return 0.0;
    }

    let tf = f64::from(tf.min(doc_len));
    let doc_len = f64::from(doc_len);
    let p_corpus = f64::from(p_corpus.clamp(0.0, 1.0));
    match smoothing {
        SmoothingMethod::JelinekMercer { lambda } => {
            if !lambda.is_finite() {
                return 0.0;
            }
            let lambda = f64::from(lambda.clamp(0.0, 1.0));
            let p_doc = if doc_len > 0.0 { tf / doc_len } else { 0.0 };
            (lambda * p_doc + (1.0 - lambda) * p_corpus) as f32
        }
        SmoothingMethod::Dirichlet { mu } => {
            if !mu.is_finite() {
                return 0.0;
            }
            let mu = f64::from(mu.max(0.0));
            let denom = doc_len + mu;
            if denom > 0.0 {
                ((tf + mu * p_corpus) / denom) as f32
            } else {
                0.0
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bm25_tf_zero_tf_is_zero() {
        let v = bm25_tf(0.0, 100.0, 100.0, 1.2, 0.75);
        assert_eq!(v, 0.0);
    }

    #[test]
    fn bm25_idf_plus1_is_non_negative() {
        assert_eq!(bm25_idf_plus1(0, 10), 0.0);
        assert_eq!(bm25_idf_plus1(10, 0), 0.0);
        assert!(bm25_idf_plus1(1000, 10) >= 0.0);
    }

    #[test]
    fn bm25_idf_plus1_decreases_with_df() {
        let n = 1_000;
        let idf_rare = bm25_idf_plus1(n, 1);
        let idf_common = bm25_idf_plus1(n, 500);
        assert!(idf_rare > idf_common);
    }

    #[test]
    fn bm25_idf_plus1_is_finite_when_df_exceeds_n() {
        // Inconsistent inputs should not yield NaN.
        let idf = bm25_idf_plus1(10, 100);
        assert!(idf.is_finite());
        assert!(idf >= 0.0);
    }

    #[test]
    fn tf_transform_variants() {
        assert_eq!(tf_transform(0, TfVariant::Linear), 0.0);
        assert_eq!(tf_transform(3, TfVariant::Linear), 3.0);
        assert_eq!(tf_transform(0, TfVariant::LogScaled), 0.0);
        assert!(tf_transform(3, TfVariant::LogScaled) > 1.0);
    }

    #[test]
    fn idf_transform_conventions() {
        assert_eq!(idf_transform(0, 10, IdfVariant::Standard), 0.0);
        assert_eq!(idf_transform(10, 0, IdfVariant::Standard), 0.0);
        assert!(idf_transform(1000, 10, IdfVariant::Standard) > 0.0);
        assert!(idf_transform(1000, 10, IdfVariant::Smoothed) > 0.0);
    }

    #[test]
    fn lm_smoothed_p_is_bounded_for_valid_inputs() {
        let p = lm_smoothed_p(
            3.0,
            10.0,
            0.01,
            SmoothingMethod::JelinekMercer { lambda: 0.2 },
        );
        assert!(p >= 0.0);
        assert!(p <= 1.0);

        // Clamp lambda.
        let p = lm_smoothed_p(
            3.0,
            10.0,
            0.01,
            SmoothingMethod::JelinekMercer { lambda: 2.0 },
        );
        assert!(p >= 0.0);
        assert!(p <= 1.0);

        // Dirichlet mu is clamped at 0.
        let p = lm_smoothed_p(3.0, 10.0, 0.01, SmoothingMethod::Dirichlet { mu: -5.0 });
        assert!(p >= 0.0);
        assert!(p <= 1.0);
    }
}
