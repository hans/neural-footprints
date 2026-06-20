configfile: "config.yaml"


SUBTRACTIVE_REGIMES = ["confounded", "area_controlled"]
SUBTRACTIVE_MODES = ["ground_truth", "inferred"]
DEFAULT_SUBTRACTIVE_MODE = "ground_truth"


wildcard_constraints:
    regime = "confounded|area_controlled",
    mode   = "ground_truth|inferred",


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
        "figures/residual_analysis.pdf",
        "figures/sample_scenes.pdf",
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
        "figures/residual_analysis.pdf",
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


rule residual:
    input:
        scenes="data/scenes.npz",
        neural="data/neural.npz",
        encoding="outputs/encoding_results.json",
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
        residual="outputs/residual_results.json",
    output:
        "outputs/evaluation.json",
    script:
        "scripts/run_evaluate.py"


# ---------------------------------------------------------------------------
# Subtractive analysis (numerosity paradigm). Parallel pipeline; runs only
# when the user asks for `subtractive_all`. Not in `rule all`.
# ---------------------------------------------------------------------------

rule generate_numerosity_scenes:
    output:
        scenes="data/numerosity_scenes_{regime}.npz",
    script:
        "scripts/gen_scenes_numerosity.py"


rule train_cardinality:
    input:
        scenes="data/numerosity_scenes_{regime}.npz",
    output:
        model="data/cardinality_model_{regime}.pt",
        acts="data/cardinality_activations_{regime}.npz",
    script:
        "scripts/train_cardinality.py"


rule generate_neural_subtractive:
    input:
        scenes="data/numerosity_scenes_{regime}.npz",
        cardinality_acts=lambda w: (
            f"data/cardinality_activations_{w.regime}.npz"
            if w.mode == "inferred" else []
        ),
    output:
        neural="data/neural_subtractive_{regime}_{mode}.npz",
    script:
        "scripts/gen_neural_subtractive.py"


rule subtractive:
    input:
        neural="data/neural_subtractive_{regime}_{mode}.npz",
    output:
        results="outputs/subtractive_{regime}_{mode}_results.json",
        plot_data="data/subtractive_{regime}_{mode}_plot_data.npz",
    script:
        "scripts/run_subtractive.py"


rule plot_subtractive:
    input:
        expand("data/subtractive_{regime}_{{mode}}_plot_data.npz",
               regime=SUBTRACTIVE_REGIMES),
    output:
        figures=expand("figures/subtractive_{regime}_{{mode}}.pdf",
                       regime=SUBTRACTIVE_REGIMES),
        headline="figures/subtractive_headline_{mode}.pdf",
    script:
        "scripts/plot_subtractive.py"


rule evaluate_subtractive:
    input:
        results=expand(
            "outputs/subtractive_{regime}_{{mode}}_results.json",
            regime=SUBTRACTIVE_REGIMES,
        ),
    output:
        "outputs/evaluation_subtractive_{mode}.json",
    script:
        "scripts/run_evaluate_subtractive.py"


rule subtractive_all:
    input:
        expand("figures/subtractive_{regime}_{mode}.pdf",
               regime=SUBTRACTIVE_REGIMES,
               mode=[DEFAULT_SUBTRACTIVE_MODE]),
        f"figures/subtractive_headline_{DEFAULT_SUBTRACTIVE_MODE}.pdf",
        f"outputs/evaluation_subtractive_{DEFAULT_SUBTRACTIVE_MODE}.json",


rule subtractive_all_modes:
    input:
        expand("figures/subtractive_{regime}_{mode}.pdf",
               regime=SUBTRACTIVE_REGIMES,
               mode=SUBTRACTIVE_MODES),
        expand("figures/subtractive_headline_{mode}.pdf",
               mode=SUBTRACTIVE_MODES),
        expand("outputs/evaluation_subtractive_{mode}.json",
               mode=SUBTRACTIVE_MODES),
