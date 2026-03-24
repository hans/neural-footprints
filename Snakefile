configfile: "config.yaml"


rule all:
    input:
        "outputs/evaluation.json",
        "figures/encoding_analysis.png",
        "figures/rsa_analysis.png",
        "figures/dissociation.png",
        "figures/predicted_frames.png",
        "figures/pca_analysis.png",
        "figures/dynamics_analysis.png",
        "figures/sample_scenes.png",


rule calibrate:
    output:
        "outputs/bullet_k.json",
    script:
        "scripts/calibrate.py"


rule generate_scenes:
    input:
        bullet_k="outputs/bullet_k.json",
    output:
        scenes="data/scenes.npz",
        figure="figures/sample_scenes.png",
    script:
        "scripts/gen_scenes.py"


rule generate_neural:
    input:
        scenes="data/scenes.npz",
    output:
        neural="data/neural.npz",
    script:
        "scripts/gen_neural.py"


rule encoding:
    input:
        scenes="data/scenes.npz",
        neural="data/neural.npz",
    output:
        results="outputs/encoding_results.json",
        figure="figures/encoding_analysis.png",
    script:
        "scripts/run_encoding.py"


rule rsa:
    input:
        scenes="data/scenes.npz",
        neural="data/neural.npz",
    output:
        results="outputs/rsa_results.json",
        figure="figures/rsa_analysis.png",
    script:
        "scripts/run_rsa.py"


rule dissociation:
    input:
        scenes="data/scenes.npz",
        neural="data/neural.npz",
    output:
        results="outputs/dissociation_results.json",
        figure="figures/dissociation.png",
        predicted="figures/predicted_frames.png",
    script:
        "scripts/run_dissociation.py"


rule pca:
    input:
        scenes="data/scenes.npz",
        neural="data/neural.npz",
    output:
        results="outputs/pca_results.json",
        figure="figures/pca_analysis.png",
    script:
        "scripts/run_pca.py"


rule dynamics:
    input:
        scenes="data/scenes.npz",
        neural="data/neural.npz",
        encoding="outputs/encoding_results.json",
    output:
        results="outputs/dynamics_results.json",
        figure="figures/dynamics_analysis.png",
    script:
        "scripts/run_dynamics.py"


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
