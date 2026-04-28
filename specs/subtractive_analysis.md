**Spec: Subtractive Analysis & the "Loudness" of Sensory Variance**
===================================================================

**1\. Scientific Motivation: Defining the Subtractive Method**
--------------------------------------------------------------

The **Subtractive Method** (or "Cognitive Subtraction") is the foundational logic of functional brain mapping. It operates on the principle of **Pure Insertion**: the assumption that a cognitive process of interest (e.g., "counting") can be added to a task without altering the underlying baseline processes (e.g., "seeing dots").

By taking the neural activation of a **Task Condition** (Dots + Counting) and subtracting the activation of a **Control Condition** (Dots only), the scientist aims to "cancel out" the shared sensory processing and isolate the specific neural population responsible for the abstract computation.

Our goal is to show that this logic collapses when the "canceled" sensory features possess high-dimensional variance. Even if a feature is shared between conditions, the _statistical footprint_ of that sensory information is so massive that it leaves behind a "residual shimmer" that masks or mimics the abstract signal.

**2\. Finding #1: The Sensible Threshold Failure**
--------------------------------------------------

Before exploring the nuances of thresholding, we first demonstrate a "Standard Failure."

*   **The Scenario:** A scientist applies a "sensible" statistical threshold (e.g., $p < 0.05$, Bonferroni corrected) to a subtraction between high-count and low-count scenes.
    
*   **The Result:** The analysis yields a clear, statistically significant "blob" in a specific region. The scientist labels this the "Numerosity Module."
    
*   **The Ground Truth:** Our simulation reveals that the "blob" actually corresponds to the **Sensory Module** (neurons tuned to edge density or luminance). Because the sensory variance was so high, it did not "cancel out" perfectly, leaving a significant residue that the scientist mistakes for an abstract representation.
    

**3\. Finding #2: The Thresholding Trap (The "No-Win" Regime)**
---------------------------------------------------------------

Building on the first finding, we show that this isn't just a matter of the scientist being "too loose" or "too strict." There is a fundamental Goldilocks problem where **no threshold** can recover the ground truth.

*   **The Low-Threshold Regime (False Positive Flood):** If the scientist lowers the threshold to ensure they don't miss the "real" signal, the map becomes overwhelmed. The "Abstract Module" may appear, but it is buried under a mountain of sensory noise and "leaky" variance from the sensory block. The signal-to-noise ratio makes the abstract computation unidentifiable.
    
*   **The High-Threshold Regime (False Negative Silence):** If the scientist raises the threshold to be "rigorous" and eliminate noise, the compact, low-dimensional abstract signal is the first thing to disappear. Because the abstract signal is "quiet" (affecting fewer neurons or having lower weight variance), it fails to survive the strict filter that the "louder" sensory residuals easily pass.
    
*   **The Conclusion:** The statistical asymmetry between sensory and abstract features ensures that the "Abstract Module" is never the most significant feature of the map.
    

**4\. Architectural Shift: Localized Block-Mapping**
----------------------------------------------------

To produce these findings, the simulation moves from random projections to a **block-structure mapping matrix**:

*   **The Sensory Block:** A large population of neurons with high-variance weights linked to pixel-level data.
    
*   **The Abstract Block:** A smaller, compact population with lower-variance weights linked to scalar variables (like $N$).
    
*   **The Resulting Anatomy:** This creates a simulated brain with distinct "functional regions," allowing us to measure exactly how much "sensory leakage" contaminates the area the scientist is studying.
    

**5\. New Simulation Pipeline: Multi-Object Scenes**
----------------------------------------------------

To generate the necessary data for these subtractions, we require a pipeline that can procedurally generate scenes with varying object counts and properties.

*   **Primary Paradigm (Numerosity):** We will generate scenes with $N$ objects. The abstract feature is the scalar $N$; the sensory features are the sum of pixels, perimeters, and centroids.
    
*   **The Goal:** Show that $N$ is always "out-shouted" by the sum of pixels in any subtractive map, regardless of the threshold chosen.
    

**6\. Summary of Interpretation**
---------------------------------

This addition demonstrates that subtractive analysis is fundamentally biased toward **representationally expensive** features. High-dimensional sensory information is "expensive" (it takes up a lot of neural real estate and variance), while abstract knowledge is "cheap" (compact and low-dimensional).

We show that our current neural analysis tools are effectively "weighted" to find the most expensive features, leading scientists to build maps of the brain's "sensory overhead" while completely missing the "computational core."
