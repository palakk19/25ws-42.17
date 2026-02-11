**1. Baseline Phase (/scripts)**
In this phase, the focus was on data understanding and establishing a performance floor using traditional modeling techniques.

Feature Selection: I utilized Mutual Information (MI) to identify the most predictive features. 

Correlation Analysis: I conducted experimental tests to analyze model stability. This involved evaluating performance shifts when dropping highly correlated columns to reduce multi-collinearity.

Model Comparison: Various baseline models were implemented here to serve as a benchmark for the final approach.

**2. Final Approach (/final_scripts)**
Insights gathered from the baseline experiments led to the development of a more sophisticated architecture.

Autoencoder Implementation: I moved to an Autoencoder approach to handle high-dimensional noise and perform effective feature compression. This approach significantly outperformed the initial baseline models.

Optimization: This folder contains the final model architecture and the specific hyperparameter tuning results that yielded the best performance.

**3. Outputs**
The results are separated to maintain a clear history of the experimental process:

output/: Performance metrics (Accuracy, F1-Score, etc.) for all baseline model comparisons.

diabetes_final_v3/: Final results for the Autoencoder training and generated summary reports.
