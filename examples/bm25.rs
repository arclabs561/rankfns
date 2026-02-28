//! BM25 and TF-IDF scoring on a tiny corpus.

use rankfns::{
    bm25_idf_plus1, bm25_tf, idf_transform, lm_smoothed_p, tf_transform, IdfVariant,
    SmoothingMethod, TfVariant,
};

fn main() {
    // Corpus: 5 documents, average length 10 tokens.
    let n_docs: u32 = 5;
    let avg_doc_len: f32 = 10.0;

    // Term "rust" appears in 2 docs; term "the" appears in all 5.
    let df_rust: u32 = 2;
    let df_the: u32 = 5;

    // --- BM25 IDF ---
    println!("=== BM25 IDF (log(1 + (N-df+0.5)/(df+0.5))) ===");
    println!("  'rust' (df={}): {:.4}", df_rust, bm25_idf_plus1(n_docs, df_rust));
    println!("  'the'  (df={}): {:.4}", df_the, bm25_idf_plus1(n_docs, df_the));

    // --- BM25 TF normalization (k1=1.2, b=0.75) ---
    println!("\n=== BM25 TF (k1=1.2, b=0.75) ===");
    let k1 = 1.2;
    let b = 0.75;
    for &(tf, doc_len) in &[(3.0_f32, 12.0_f32), (1.0, 8.0), (0.0, 10.0)] {
        let score = bm25_tf(tf, doc_len, avg_doc_len, k1, b);
        println!("  tf={:.0}, doc_len={:.0} => {:.4}", tf, doc_len, score);
    }

    // --- Full BM25 score = IDF * TF ---
    println!("\n=== Full BM25 score (IDF * TF) ===");
    let idf = bm25_idf_plus1(n_docs, df_rust);
    let tf_score = bm25_tf(3.0, 12.0, avg_doc_len, k1, b);
    println!("  'rust' in doc (tf=3, len=12): {:.4}", idf * tf_score);

    // --- TF-IDF variants ---
    println!("\n=== TF-IDF variants ===");
    let tf_lin = tf_transform(3, TfVariant::Linear);
    let tf_log = tf_transform(3, TfVariant::LogScaled);
    let idf_std = idf_transform(n_docs, df_rust, IdfVariant::Standard);
    let idf_smooth = idf_transform(n_docs, df_rust, IdfVariant::Smoothed);
    println!("  Linear TF(3)={:.2}, LogScaled TF(3)={:.4}", tf_lin, tf_log);
    println!("  Standard IDF={:.4}, Smoothed IDF={:.4}", idf_std, idf_smooth);
    println!("  TF-IDF (linear, standard): {:.4}", tf_lin * idf_std);

    // --- Language model smoothing ---
    println!("\n=== Language model P(t|D) ===");
    let p_corpus = 0.01; // corpus probability of "rust"
    let p_jm = lm_smoothed_p(3.0, 12.0, p_corpus, SmoothingMethod::JelinekMercer { lambda: 0.7 });
    let p_dir = lm_smoothed_p(3.0, 12.0, p_corpus, SmoothingMethod::Dirichlet { mu: 1000.0 });
    println!("  Jelinek-Mercer (lambda=0.7): {:.4}", p_jm);
    println!("  Dirichlet      (mu=1000):    {:.4}", p_dir);
}
