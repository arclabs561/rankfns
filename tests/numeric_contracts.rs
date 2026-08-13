use rankfns::{bm25_idf_plus1, bm25_tf, idf_transform, lm_smoothed_p, IdfVariant, SmoothingMethod};

const TOL: f32 = 1e-6;

#[test]
fn canonical_two_document_bm25_matches_lucene_formula() {
    let idf = bm25_idf_plus1(2, 1);
    let tf = bm25_tf(1.0, 10.0, 10.0, 1.2, 0.75);
    assert!((idf - 2.0_f32.ln()).abs() < TOL);
    assert!((tf - 1.0).abs() < TOL);
    assert!((idf * tf - 2.0_f32.ln()).abs() < TOL);
}

#[test]
fn smoothed_idf_uses_one_invalid_stat_policy() {
    assert_eq!(
        idf_transform(10, 100, IdfVariant::Smoothed),
        bm25_idf_plus1(10, 100)
    );
}

#[test]
fn bm25_tf_returns_zero_for_nonfinite_inputs() {
    let cases = [
        (f32::NAN, 10.0, 10.0, 1.2, 0.75),
        (f32::INFINITY, 10.0, 10.0, 1.2, 0.75),
        (1.0, f32::NAN, 10.0, 1.2, 0.75),
        (1.0, 10.0, f32::INFINITY, 1.2, 0.75),
        (1.0, 10.0, 10.0, f32::NAN, 0.75),
        (1.0, 10.0, 10.0, 1.2, f32::INFINITY),
    ];
    for (tf, dl, avgdl, k1, b) in cases {
        assert_eq!(bm25_tf(tf, dl, avgdl, k1, b), 0.0);
    }
}

#[test]
fn bm25_tf_is_finite_and_monotone_on_valid_domain() {
    for avgdl in [1.0, 10.0, 1_000.0] {
        let mut previous_tf_score = 0.0;
        for tf in [0.0, 1.0, 2.0, 10.0, 1_000_000.0, f32::MAX] {
            let score = bm25_tf(tf, avgdl, avgdl, 1.2, 0.75);
            assert!(score.is_finite() && score >= previous_tf_score);
            previous_tf_score = score;
        }

        let short = bm25_tf(3.0, avgdl / 2.0, avgdl, 1.2, 0.75);
        let long = bm25_tf(3.0, avgdl * 2.0, avgdl, 1.2, 0.75);
        assert!(short > long);
        assert_eq!(
            bm25_tf(3.0, avgdl / 2.0, avgdl, 1.2, 0.0),
            bm25_tf(3.0, avgdl * 2.0, avgdl, 1.2, 0.0)
        );
    }
}

#[test]
fn lm_invalid_inputs_return_zero_and_valid_outputs_are_probabilities() {
    for smoothing in [
        SmoothingMethod::JelinekMercer { lambda: 0.2 },
        SmoothingMethod::Dirichlet { mu: 100.0 },
    ] {
        for (tf, dl, pc) in [
            (f32::NAN, 10.0, 0.1),
            (1.0, f32::INFINITY, 0.1),
            (1.0, 10.0, f32::NAN),
            (-1.0, 10.0, 0.1),
            (1.0, -10.0, 0.1),
        ] {
            assert_eq!(lm_smoothed_p(tf, dl, pc, smoothing), 0.0);
        }

        for dl in [0.0, 1.0, 10.0, 1_000.0] {
            for tf in [0.0, dl / 2.0, dl, dl + 1.0] {
                for pc in [0.0, 0.1, 1.0] {
                    let p = lm_smoothed_p(tf, dl, pc, smoothing);
                    assert!(p.is_finite() && (0.0..=1.0).contains(&p));
                }
            }
        }
    }

    for smoothing in [
        SmoothingMethod::JelinekMercer { lambda: f32::NAN },
        SmoothingMethod::JelinekMercer {
            lambda: f32::INFINITY,
        },
        SmoothingMethod::Dirichlet { mu: f32::NAN },
        SmoothingMethod::Dirichlet { mu: f32::INFINITY },
    ] {
        assert_eq!(lm_smoothed_p(1.0, 10.0, 0.1, smoothing), 0.0);
    }
}

#[test]
fn jelinek_mercer_lambda_is_document_weight() {
    let p_doc = 0.3;
    let p_corpus = 0.01;
    assert_eq!(
        lm_smoothed_p(
            3.0,
            10.0,
            p_corpus,
            SmoothingMethod::JelinekMercer { lambda: 0.0 }
        ),
        p_corpus
    );
    assert!(
        (lm_smoothed_p(
            3.0,
            10.0,
            p_corpus,
            SmoothingMethod::JelinekMercer { lambda: 1.0 }
        ) - p_doc)
            .abs()
            < TOL
    );
}

#[test]
fn dirichlet_matches_its_convex_mixture_identity() {
    let tf = 3.0;
    let doc_len = 10.0;
    let p_corpus = 0.01;
    let mu = 100.0;
    let document_weight = doc_len / (doc_len + mu);
    let expected = document_weight * (tf / doc_len) + (1.0 - document_weight) * p_corpus;
    let actual = lm_smoothed_p(tf, doc_len, p_corpus, SmoothingMethod::Dirichlet { mu });
    assert!((actual - expected).abs() < TOL);
}
