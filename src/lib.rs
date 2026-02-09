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
/// - `avg_doc_len` and the denominator are clamped away from 0.
pub fn bm25_tf(tf: f32, doc_len: f32, avg_doc_len: f32, k1: f32, b: f32) -> f32 {
    if tf <= 0.0 {
        return 0.0;
    }
    let avg = avg_doc_len.max(1e-9);
    let k1 = k1.max(0.0);
    let b = b.clamp(0.0, 1.0);
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
pub fn lm_smoothed_p(tf: f32, doc_len: f32, p_corpus: f32, smoothing: SmoothingMethod) -> f32 {
    let p_corpus = p_corpus.clamp(0.0, 1.0);
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
