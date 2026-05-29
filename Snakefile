configfile: "config.yaml"

_ALL_FIGURES = [
    "figures/{norm}/encoding_analysis.pdf",
    "figures/{norm}/rsa_analysis.pdf",
    "figures/{norm}/dissociation.pdf",
    "figures/{norm}/dissociation_combined.pdf",
    "figures/{norm}/predicted_frames.pdf",
    "figures/{norm}/predicted_frames_compact.pdf",
    "figures/{norm}/pca_analysis.pdf",
    "figures/{norm}/pca_variance_decoding.pdf",
    "figures/{norm}/dynamics_analysis.pdf",
    "figures/{norm}/predictive_processing.pdf",
    "figures/{norm}/pp_frames.pdf",
    "figures/{norm}/residual_analysis.pdf",
    "figures/{norm}/sample_scenes.pdf",
    "figures/{norm}/p_block_contribution.pdf",
]


rule all:
    input:
        expand("outputs/{norm}/evaluation.json", norm=config["block_norms"]),
        expand(_ALL_FIGURES, norm=config["block_norms"]),
        expand("paper/{norm}_results_macros.tex", norm=config["block_norms"]),
        "figures/p_block_contribution_compare.pdf",


rule figures:
    input:
        expand(_ALL_FIGURES, norm=config["block_norms"]),


rule norm_all:
    input:
        expand("outputs/{norm}/evaluation.json", norm=[config["block_norm"]]),
        expand(_ALL_FIGURES, norm=[config["block_norm"]]),
        expand("paper/{norm}_results_macros.tex", norm=[config["block_norm"]]),


# ---------------------------------------------------------------------------
# Data generation (norm-independent: shared across all norms)
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


rule forward_render:
    input:
        scenes="data/scenes.npz",
        pp_activations="data/pp_activations.npz",
    output:
        forward_renders="data/forward_renders.npz",
    script:
        "scripts/gen_forward_renders.py"


rule generate_neural:
    input:
        scenes="data/scenes.npz",
        pp_activations="data/pp_activations.npz",
        forward_renders="data/forward_renders.npz",
    output:
        neural="data/{norm}/neural.npz",
    params:
        block_norm=lambda wildcards: wildcards.norm,
    script:
        "scripts/gen_neural.py"


# ---------------------------------------------------------------------------
# Analysis (computation only — no figures)
# ---------------------------------------------------------------------------

rule predictive_processing:
    input:
        scenes="data/scenes.npz",
        neural="data/{norm}/neural.npz",
        model="data/inverse_model.pt",
    output:
        results="outputs/{norm}/pp_results.json",
        inferred="data/{norm}/inferred_physics.npz",
        plot_data="data/{norm}/pp_plot_data.npz",
    script:
        "scripts/run_pp.py"


rule encoding:
    input:
        scenes="data/scenes.npz",
        neural="data/{norm}/neural.npz",
        forward_renders="data/forward_renders.npz",
    output:
        results="outputs/{norm}/encoding_results.json",
        encoder="data/{norm}/encoder.joblib",
        plot_data="data/{norm}/encoding_plot_data.npz",
    script:
        "scripts/run_encoding.py"


rule rsa:
    input:
        scenes="data/scenes.npz",
        neural="data/{norm}/neural.npz",
        forward_renders="data/forward_renders.npz",
    output:
        results="outputs/{norm}/rsa_results.json",
        plot_data="data/{norm}/rsa_plot_data.npz",
    script:
        "scripts/run_rsa.py"


rule dissociation:
    input:
        scenes="data/scenes.npz",
        neural="data/{norm}/neural.npz",
        encoder="data/{norm}/encoder.joblib",
        forward_renders="data/forward_renders.npz",
    output:
        results="outputs/{norm}/dissociation_results.json",
        plot_data="data/{norm}/dissociation_plot_data.npz",
    script:
        "scripts/run_dissociation.py"


rule pca:
    input:
        scenes="data/scenes.npz",
        neural="data/{norm}/neural.npz",
    output:
        results="outputs/{norm}/pca_results.json",
        plot_data="data/{norm}/pca_plot_data.npz",
    script:
        "scripts/run_pca.py"


rule dynamics:
    input:
        scenes="data/scenes.npz",
        neural="data/{norm}/neural.npz",
        encoding="outputs/{norm}/encoding_results.json",
        encoder="data/{norm}/encoder.joblib",
        inferred="data/{norm}/inferred_physics.npz",
    output:
        results="outputs/{norm}/dynamics_results.json",
        plot_data="data/{norm}/dynamics_plot_data.npz",
    script:
        "scripts/run_dynamics.py"


rule residual:
    input:
        scenes="data/scenes.npz",
        neural="data/{norm}/neural.npz",
        encoding="outputs/{norm}/encoding_results.json",
        forward_renders="data/forward_renders.npz",
    output:
        results="outputs/{norm}/residual_results.json",
        plot_data="data/{norm}/residual_plot_data.npz",
    script:
        "scripts/run_residual.py"


# ---------------------------------------------------------------------------
# Plotting (fast — only reads cached plot_data NPZ files)
# ---------------------------------------------------------------------------

rule plot_scenes:
    input:
        scenes="data/scenes.npz",
        forward_renders="data/forward_renders.npz",
    output:
        figure="figures/{norm}/sample_scenes.pdf",
    script:
        "scripts/plot_scenes.py"


rule plot_encoding:
    input:
        plot_data="data/{norm}/encoding_plot_data.npz",
    output:
        figure="figures/{norm}/encoding_analysis.pdf",
    script:
        "scripts/plot_encoding.py"


rule plot_rsa:
    input:
        plot_data="data/{norm}/rsa_plot_data.npz",
    output:
        figure="figures/{norm}/rsa_analysis.pdf",
    script:
        "scripts/plot_rsa.py"


rule plot_dissociation:
    input:
        plot_data="data/{norm}/dissociation_plot_data.npz",
    output:
        figure="figures/{norm}/dissociation.pdf",
        combined="figures/{norm}/dissociation_combined.pdf",
        predicted="figures/{norm}/predicted_frames.pdf",
        predicted_compact="figures/{norm}/predicted_frames_compact.pdf",
    script:
        "scripts/plot_dissociation.py"


rule plot_pca:
    input:
        plot_data="data/{norm}/pca_plot_data.npz",
    output:
        figure="figures/{norm}/pca_analysis.pdf",
        overlay="figures/{norm}/pca_variance_decoding.pdf",
    script:
        "scripts/plot_pca.py"


rule plot_dynamics:
    input:
        plot_data="data/{norm}/dynamics_plot_data.npz",
    output:
        figure="figures/{norm}/dynamics_analysis.pdf",
    script:
        "scripts/plot_dynamics.py"


rule plot_pp:
    input:
        plot_data="data/{norm}/pp_plot_data.npz",
        forward_renders="data/forward_renders.npz",
        scenes="data/scenes.npz",
    output:
        figure="figures/{norm}/predictive_processing.pdf",
        frames="figures/{norm}/pp_frames.pdf",
    script:
        "scripts/plot_pp.py"


rule plot_residual:
    input:
        plot_data="data/{norm}/residual_plot_data.npz",
    output:
        figure="figures/{norm}/residual_analysis.pdf",
    script:
        "scripts/plot_residual.py"


# ---------------------------------------------------------------------------
# Evaluation & paper macros
# ---------------------------------------------------------------------------

rule paper_macros:
    input:
        encoding="outputs/{norm}/encoding_results.json",
        rsa="outputs/{norm}/rsa_results.json",
        dynamics="outputs/{norm}/dynamics_results.json",
        pca="outputs/{norm}/pca_results.json",
        evaluation="outputs/{norm}/evaluation.json",
    output:
        "paper/{norm}_results_macros.tex",
    script:
        "scripts/gen_macros.py"


rule evaluate:
    input:
        encoding="outputs/{norm}/encoding_results.json",
        rsa="outputs/{norm}/rsa_results.json",
        dissociation="outputs/{norm}/dissociation_results.json",
        dynamics="outputs/{norm}/dynamics_results.json",
        residual="outputs/{norm}/residual_results.json",
    output:
        "outputs/{norm}/evaluation.json",
    script:
        "scripts/run_evaluate.py"


# ---------------------------------------------------------------------------
# Per-block P contribution diagnostic (scientific analysis, not evaluation)
# ---------------------------------------------------------------------------

rule p_block_contribution:
    input:
        scenes="data/scenes.npz",
        neural="data/{norm}/neural.npz",
        pp_activations="data/pp_activations.npz",
        forward_renders="data/forward_renders.npz",
    output:
        results="outputs/{norm}/p_block_contribution.json",
        plot_data="data/{norm}/p_block_plot_data.npz",
    script:
        "scripts/run_p_block_contribution.py"


rule plot_p_block_contribution:
    input:
        plot_data="data/{norm}/p_block_plot_data.npz",
    output:
        figure="figures/{norm}/p_block_contribution.pdf",
    script:
        "scripts/plot_p_block_contribution.py"


rule plot_p_block_compare:
    input:
        zscore="data/zscore/p_block_plot_data.npz",
        truncated_svd="data/truncated_svd/p_block_plot_data.npz",
    output:
        figure="figures/p_block_contribution_compare.pdf",
    script:
        "scripts/plot_p_block_compare.py"
