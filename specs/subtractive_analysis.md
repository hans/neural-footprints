This specification document outlines a proposed "Subtractive Analysis" simulation. The goal is to demonstrate that the most common method in cognitive neuroscience—comparing two conditions to isolate a cognitive process—systematically fails when the target representation is low-dimensional (abstract) and the confounding representation is high-dimensional (sensory).

**Scientific Motivation**
-------------------------

The "Subtractive Method" assumes that by subtracting a control condition from a task condition, we cancel out shared "noise" and isolate the "signal." However, if the "signal" is a compact, abstract variable (like the number 5) and the "noise" is a high-variance sensory feature (like the distribution of 500 pixels), the subtraction will be dominated by sensory variance.

We aim to show that a "Sensory-Only" model can produce a brain map that looks identical to a "Full" model, leading a scientist to believe they have found an abstract module where none exists.

**Option 1: Motion (Velocity vs. Flicker)**
-------------------------------------------

*   **The Scientist’s Task:** Identify a "Motion Region" (e.g., Area MT+).
    
*   **The Subtraction:** $\\text{Moving Dots} - \\text{Stationary Dots}$.
    
*   **The Target (Abstract):** A velocity vector (the physical direction and speed of an object).
    
*   **The Confound (Sensory):** Temporal Frequency/Flicker (the rate of change in pixel intensity at a specific location).
    
*   **Interpretation:** Because moving dots create massive amounts of local pixel variance (flicker) compared to static dots, the scientist identifies a "Motion Region" that is actually just a "High-Variance Change Detector."
    

**Option 2: Numerosity (Cardinality vs. Spatial Sums)**
-------------------------------------------------------

*   **The Scientist’s Task:** Identify a "Number Map" in the parietal lobe (IPS).
    
*   **The Subtraction:** $\\text{High Count (N=12)} - \\text{Low Count (N=3)}$.
    
*   **The Target (Abstract):** Cardinality (the abstract concept of "twelve-ness").
    
*   **The Confound (Sensory):** Total Surface Area, Edge Density, and Luminance.
    
*   **Interpretation:** A model that only knows "how much ink is on the screen" (high-dimensional sensory flux) will pass the subtraction test with a higher $z$-score than a model that actually "counts." The scientist mistakes sensory accumulation for abstract math.
    

**Option 3: Structural Coherence (Intact vs. Scrambled)**
---------------------------------------------------------

*   **The Scientist’s Task:** Identify an "Object-Selective Region" (e.g., LOC).
    
*   **The Subtraction:** $\\text{Intact Photos} - \\text{Scrambled Pixels}$.
    
*   **The Target (Abstract):** Structural coherence and object identity (the "concept" of a chair).
    
*   **The Confound (Sensory):** Low-level spatial correlations and power spectra.
    
*   **Interpretation:** Scrambling an image radically alters its high-dimensional statistical fingerprint. The scientist thinks they have found a "Chair Detector," but they have actually found a "Non-Random Noise Detector."
    

**Strategic Recommendation: Starting with Numerosity**
------------------------------------------------------

We recommend the **Numerosity** paradigm for the initial implementation for several reasons:

1.  **Mathematical Clarity:** The abstract feature is a pure scalar (a single number). This provides the starkest possible contrast to the high-dimensional sensory input (the thousands of pixels/coordinates required to render those dots). It is the "cleanest" example of statistical asymmetry.
    
2.  **Well-Documented Confounds:** The "Numerosity vs. Continuous Variables" debate is a classic in the literature. We don't have to invent the confounds; we can use established ones like total perimeter and occupancy.
    
3.  **Provocative Interpretation:** Showing that a "Number Map"—one of the most cited results in functional brain mapping—could be a statistical artifact of sensory "loudness" provides the most compelling narrative for the paper. It perfectly illustrates our core thesis: **variance-based methods find the "commotion" of the sensory signal, not the "computation" of the abstract one.**
    

### **Success Criteria for the Simulation**

*   **The False Positive:** A "Sensory-Only" model (with zero knowledge of number/motion) generates a statistically significant "Brain Map" that mimics experimental data.
    
*   **The Failure of Fit:** Standard encoding models fail to show a significant improvement when the "True" abstract variable is added to the "Sensory-Only" model, because the sensory variance masks the abstract signal.
    
*   **The Ground Truth Gap:** We show that while the scientist's map is "significant," it is fundamentally unrelated to the causally relevant variables in the simulation.
