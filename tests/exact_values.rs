//! Hand-computed exact values for the IR scoring kernels.
//!
//! The crate's own tests check signs, monotonicity, and bounds but never the
//! actual numbers, so a variant/off-by-one bug (b vs k1 placement, the +0.5 in
//! the IDF, the smoothing interpolation) would pass them. These pin the closed
//! forms to values computed by hand.

use rankfns::{
    bm25_idf_plus1, bm25_tf, idf_transform, lm_smoothed_p, tf_transform, IdfVariant,
    SmoothingMethod, TfVariant,
};

const TOL: f32 = 1e-4;

#[test]
fn bm25_tf_exact() {
    // dl == avgdl => length term = 1-b+b*1 = 1; denom = tf + k1.
    // 2*(1.2+1) / (2 + 1.2) = 4.4/3.2 = 1.375.
    assert!((bm25_tf(2.0, 100.0, 100.0, 1.2, 0.75) - 1.375).abs() < TOL);
    // b = 0 => length-independent (same value at a different doc_len).
    assert!((bm25_tf(2.0, 50.0, 100.0, 1.2, 0.0) - 1.375).abs() < TOL);
    // Saturates to k1+1 as tf -> infinity.
    assert!((bm25_tf(1e9, 100.0, 100.0, 1.2, 0.75) - 2.2).abs() < 1e-2);
}

#[test]
fn bm25_idf_plus1_exact() {
    // ln( (100-10+0.5)/(10+0.5) + 1 ) = ln(90.5/10.5 + 1) = ln(9.619047) = 2.263839.
    assert!((bm25_idf_plus1(100, 10) - 2.263_839).abs() < TOL);
}

#[test]
fn idf_transform_exact() {
    // Standard: ln(N/df) = ln(1000/10) = ln(100).
    assert!((idf_transform(1000, 10, IdfVariant::Standard) - 100.0_f32.ln()).abs() < TOL);
    // Smoothed equals the bm25_idf_plus1 formula.
    assert!(
        (idf_transform(100, 10, IdfVariant::Smoothed) - bm25_idf_plus1(100, 10)).abs() < TOL
    );
}

#[test]
fn tf_transform_exact() {
    assert!((tf_transform(3, TfVariant::Linear) - 3.0).abs() < TOL);
    // 1 + ln(3) = 2.098612.
    assert!((tf_transform(3, TfVariant::LogScaled) - 2.098_612).abs() < TOL);
}

#[test]
fn lm_smoothed_p_exact() {
    // Dirichlet: (tf + mu*p_c)/(dl + mu) = (3 + 100*0.01)/(10 + 100) = 4/110 = 0.0363636.
    let d = lm_smoothed_p(3.0, 10.0, 0.01, SmoothingMethod::Dirichlet { mu: 100.0 });
    assert!((d - 0.036_363_6).abs() < TOL);
    // Dirichlet with mu -> 0 reduces to the MLE tf/dl = 0.3.
    let mle = lm_smoothed_p(3.0, 10.0, 0.01, SmoothingMethod::Dirichlet { mu: 0.0 });
    assert!((mle - 0.3).abs() < TOL);
    // Jelinek-Mercer: lam*(tf/dl) + (1-lam)*p_c = 0.2*0.3 + 0.8*0.01 = 0.068.
    let jm = lm_smoothed_p(3.0, 10.0, 0.01, SmoothingMethod::JelinekMercer { lambda: 0.2 });
    assert!((jm - 0.068).abs() < TOL);
}
