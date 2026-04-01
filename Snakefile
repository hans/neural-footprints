configfile: "config.yaml"


rule all:
    input:
        "outputs/evaluation.json",
        "figures/encoding_analysis.pdf",
        "figures/rsa_analysis.pdf",
        "figures/dissociation.pdf",
        "figures/predicted_frames.pdf",
        "figures/pca_analysis.pdf",
        "figures/pca_variance_decoding.pdf",
        "figures/dynamics_analysis.pdf",
        "figures/sample_scenes.pdf",
        "paper/results_macros.tex",


rule figures:
    input:
        "figures/encoding_analysis.pdf",
        "figures/rsa_analysis.pdf",
        "figures/dissociation.pdf",
        "figures/predicted_frames.pdf",
        "figures/pca_analysis.pdf",
        "figures/pca_variance_decoding.pdf",
        "figures/dynamics_analysis.pdf",
        "figures/sample_scenes.pdf",


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

rule generate_scenes:
    output:
        scenes="data/scenes.npz",
    script:
        "scripts/gen_scenes.py"


rule generate_neural:
    input:
        scenes="data/scenes.npz",
    output:
        neural="data/neural.npz",
    script:
        "scripts/gen_neural.py"


# ---------------------------------------------------------------------------
# Analysis (computation only — no figures)
# ---------------------------------------------------------------------------

rule encoding:
    input:
        scenes="data/scenes.npz",
        neural="data/neural.npz",
    output:
        results="outputs/encoding_results.json",
        encoder="data/encoder.joblib",
        plot_data="data/encoding_plot_data.npz",
    script:
        "scripts/run_encoding.py"


rule rsa:
    input:
        scenes="data/scenes.npz",
        neural="data/neural.npz",
    output:
        results="outputs/rsa_results.json",
        plot_data="data/rsa_plot_data.npz",
    script:
        "scripts/run_rsa.py"


rule dissociation:
    input:
        scenes="data/scenes.npz",
        neural="data/neural.npz",
        encoder="data/encoder.joblib",
    output:
        results="outputs/dissociation_results.json",
        plot_data="data/dissociation_plot_data.npz",
    script:
        "scripts/run_dissociation.py"


rule pca:
    input:
        scenes="data/scenes.npz",
        neural="data/neural.npz",
    output:
        results="outputs/pca_results.json",
        plot_data="data/pca_plot_data.npz",
    script:
        "scripts/run_pca.py"


rule dynamics:
    input:
        scenes="data/scenes.npz",
        neural="data/neural.npz",
        encoding="outputs/encoding_results.json",
        encoder="data/encoder.joblib",
    output:
        results="outputs/dynamics_results.json",
        plot_data="data/dynamics_plot_data.npz",
    script:
        "scripts/run_dynamics.py"


# ---------------------------------------------------------------------------
# Plotting (fast — only reads cached plot_data NPZ files)
# ---------------------------------------------------------------------------

rule plot_scenes:
    input:
        scenes="data/scenes.npz",
    output:
        figure="figures/sample_scenes.pdf",
    script:
        "scripts/plot_scenes.py"


rule plot_encoding:
    input:
        plot_data="data/encoding_plot_data.npz",
    output:
        figure="figures/encoding_analysis.pdf",
    script:
        "scripts/plot_encoding.py"


rule plot_rsa:
    input:
        plot_data="data/rsa_plot_data.npz",
    output:
        figure="figures/rsa_analysis.pdf",
    script:
        "scripts/plot_rsa.py"


rule plot_dissociation:
    input:
        plot_data="data/dissociation_plot_data.npz",
    output:
        figure="figures/dissociation.pdf",
        predicted="figures/predicted_frames.pdf",
    script:
        "scripts/plot_dissociation.py"


rule plot_pca:
    input:
        plot_data="data/pca_plot_data.npz",
    output:
        figure="figures/pca_analysis.pdf",
        overlay="figures/pca_variance_decoding.pdf",
    script:
        "scripts/plot_pca.py"


rule plot_dynamics:
    input:
        plot_data="data/dynamics_plot_data.npz",
    output:
        figure="figures/dynamics_analysis.pdf",
    script:
        "scripts/plot_dynamics.py"


# ---------------------------------------------------------------------------
# Evaluation & paper macros
# ---------------------------------------------------------------------------

rule paper_macros:
    input:
        encoding="outputs/encoding_results.json",
        rsa="outputs/rsa_results.json",
        dynamics="outputs/dynamics_results.json",
        pca="outputs/pca_results.json",
        evaluation="outputs/evaluation.json",
    output:
        "paper/results_macros.tex",
    script:
        "scripts/gen_macros.py"


rule evaluate:
    input:
        encoding="outputs/encoding_results.json",
        rsa="outputs/rsa_results.json",
        dissociation="outputs/dissociation_results.json",
        dynamics="outputs/dynamics_results.json",
    output:
        "outputs/evaluation.json",
    script:
        "scripts/run_evaluate.py"
