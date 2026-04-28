**Spec: Subtractive Analysis & the "Loudness" of Sensory Variance**
===================================================================

**1\. Scientific Motivation: The Subtractive Method**
-----------------------------------------------------

The **Subtractive Method** (or "Cognitive Subtraction") is the bedrock of functional brain mapping. It operates on the principle of **Pure Insertion**: the assumption that a cognitive process of interest can be added to a task without altering the underlying baseline processes.

By taking the neural activation of a **Task Condition** and subtracting the activation of a **Control Condition**, scientists aim to "cancel out" shared sensory processing and isolate the specific neural population responsible for the abstract computation. Our goal is to show that this logic collapses when the "canceled" sensory features possess high-dimensional variance. The statistical footprint of sensory information is so massive that it leaves behind a "residual shimmer" that masks or mimics the abstract signal.

**2\. Experimental Paradigms (Three Comparison Options)**
---------------------------------------------------------

To demonstrate the breadth of this failure, the simulation pipeline must support three classic "flavors" of vision neuroscience subtractions:

### **Option A: Numerosity (The "Number Sense")**

*   **The Subtraction:** High Count (e.g., $N=12$) $-$ Low Count (e.g., $N=3$).
    
*   **The Target (Abstract):** Cardinality (the scalar concept of "twelve-ness").
    
*   **The Confound (Sensory):** Cumulative surface area, edge density, and total luminance.
    
*   **Motivation:** This is the cleanest mathematical gap—a 1D scalar vs. a thousand-dimensional sensory footprint.
    

### **Option B: Motion (Velocity vs. Flicker)**

*   **The Subtraction:** $N$ Moving Objects $-$ $N$ Stationary Objects.
    
*   **The Target (Abstract):** Velocity vectors and physical trajectories.
    
*   **The Confound (Sensory):** Temporal flicker and pixel-level variance.
    
*   **Motivation:** Mirrors the classic Area MT+ localization debate; distinguishes between "seeing change" and "computing motion."
    

### **Option C: Structural Coherence (Intact vs. Scrambled)**

*   **The Subtraction:** Intact Objects $-$ Scrambled Pixels.
    
*   **The Target (Abstract):** Structural integrity and object identity.
    
*   **The Confound (Sensory):** Low-level spatial correlations and power spectra.
    
*   **Motivation:** Targets the "Object-Selective" regions (LOC); shows that "intactness" is often confounded by the statistical regularity of the image.
    

**3\. Implementation: Localized Block-Mapping**
-----------------------------------------------

To move beyond global neural states and create a "Brain Map," the simulation uses a **block-structure mapping matrix**:

*   **The Sensory Block:** A large population of neurons with high-variance weights linked to pixel-level data.
    
*   **The Abstract Block:** A smaller, compact population with lower-variance weights linked to the abstract variable.
    
*   **Scientific Utility:** This allows for a ground-truth "Gold Standard." We can measure exactly how much "sensory leakage" from the Sensory Block contaminates the "Abstract Block" during analysis.
    

**4\. Proposed Findings**
-------------------------

### **Finding #1: The Sensible Threshold Failure**

We first demonstrate that at a "standard" statistical threshold (e.g., $p < 0.05$, corrected), the scientist consistently misidentifies the source of the neural activity.

*   **The Error:** The analysis yields a significant "blob." However, the simulation reveals this blob corresponds to the **Sensory Block**. Because the sensory variance is so high, it does not "cancel out" perfectly, leaving a significant residue that mimics a functional module.
    

### **Finding #2: The Thresholding Trap (The "No-Win" Regime)**

We then show that no choice of threshold can recover the truth.

*   **Low Threshold (False Positive Flood):** The "Abstract Block" is technically active, but it is buried under a mountain of sensory leakage. The signal-to-noise ratio is too low to identify the computation.
    
*   **High Threshold (False Negative Silence):** As the scientist raises the threshold to be "rigorous," the quiet, compact abstract signal is the first to be deleted. The high-variance sensory residuals are the only features loud enough to survive.
    
*   **The Conclusion:** There is **no threshold** where the Abstract Block is the most significant feature of the map.
    

**5\. Strategic Recommendation: Starting with Numerosity**
----------------------------------------------------------

The **Numerosity** paradigm will be the primary implementation target.

1.  **Starkest Asymmetry:** The gap between a scalar ($N$) and its sensory footprint is the most extreme.
    
2.  **Historical Impact:** The "Number Map" in the IPS is a high-profile target for demonstrating how "sensory overhead" can be mistaken for "computational core."
    
3.  **Simulated Simplicity:** $N$ spheres are easier to procedurally generate and control than complex motion physics or scrambled textures.
