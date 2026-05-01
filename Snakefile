configfile: "config.yaml"


rule all:
    input:
        "outputs/evaluation.json",
        "figures/encoding_analysis.pdf",
        "figures/rsa_analysis.pdf",
        "figures/dissociation.pdf",
        "figures/dissociation_combined.pdf",
        "figures/predicted_frames.pdf",
        "figures/predicted_frames_compact.pdf",
        "figures/pca_analysis.pdf",
        "figures/pca_variance_decoding.pdf",
        "figures/dynamics_analysis.pdf",
        "figures/predictive_processing.pdf",
        "figures/pp_frames.pdf",
        "figures/sample_scenes.pdf",
        "figures/residual_analysis.pdf",
        "paper/results_macros.tex",


rule figures:
    input:
        "figures/encoding_analysis.pdf",
        "figures/rsa_analysis.pdf",
        "figures/dissociation.pdf",
        "figures/dissociation_combined.pdf",
        "figures/predicted_frames.pdf",
        "figures/predicted_frames_compact.pdf",
        "figures/pca_analysis.pdf",
        "figures/pca_variance_decoding.pdf",
        "figures/dynamics_analysis.pdf",
        "figures/predictive_processing.pdf",
        "figures/pp_frames.pdf",
        "figures/sample_scenes.pdf",
        "figures/residual_analysis.pdf",


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

rule generate_scenes:
    output:
        scenes="data/scenes.npz",
    script:
        "scripts/gen_scenes.py"


rule train_pp_for_neural:
    input:
        scenes="data/scenes.npz",
    output:
        model="data/inverse_model.pt",
        pp_acts="data/pp_activations.npz",
    script:
        "scripts/train_pp_for_neural.py"


rule generate_neural:
    input:
        scenes="data/scenes.npz",
        pp_activations="data/pp_activations.npz",
    output:
        neural="data/neural.npz",
    script:
        "scripts/gen_neural.py"


# ---------------------------------------------------------------------------
# Analysis (computation only — no figures)
# ---------------------------------------------------------------------------

rule predictive_processing:
    input:
        scenes="data/scenes.npz",
        neural="data/neural.npz",
        model="data/inverse_model.pt",
    output:
        results="outputs/pp_results.json",
        inferred="data/inferred_physics.npz",
        plot_data="data/pp_plot_data.npz",
    script:
        "scripts/run_pp.py"


rule encoding:
    input:
        scenes="data/scenes.npz",
        neural="data/neural.npz",
        inferred="data/inferred_physics.npz",
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
        inferred="data/inferred_physics.npz",
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
        pp_results="outputs/pp_results.json",
        pp_plot_data="data/pp_plot_data.npz",
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


rule residual:
    input:
        scenes="data/scenes.npz",
        neural="data/neural.npz",
        inferred="data/inferred_physics.npz",
    output:
        results="outputs/residual_results.json",
        plot_data="data/residual_plot_data.npz",
    script:
        "scripts/run_residual.py"


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
        combined="figures/dissociation_combined.pdf",
        predicted="figures/predicted_frames.pdf",
        predicted_compact="figures/predicted_frames_compact.pdf",
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


rule plot_pp:
    input:
        plot_data="data/pp_plot_data.npz",
    output:
        figure="figures/predictive_processing.pdf",
        frames="figures/pp_frames.pdf",
    script:
        "scripts/plot_pp.py"


rule plot_residual:
    input:
        plot_data="data/residual_plot_data.npz",
    output:
        figure="figures/residual_analysis.pdf",
    script:
        "scripts/plot_residual.py"


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
        pp_results="outputs/pp_results.json",
        residual="outputs/residual_results.json",
    output:
        "outputs/evaluation.json",
    script:
        "scripts/run_evaluate.py"
