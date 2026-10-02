# Viva Defense Preparation & Technical Q&A Guide
## Multimodal Financial Fraud / Risk Detector

This guide prepares the project author for academic defense, oral examination, and technical code review. Each question is paired with a concise **Summary Answer** followed by an **In-Depth Technical Defense**.

---

### Section 1: Deep Neural Architecture & Entity Embeddings

#### Q1: Why did you use categorical entity embeddings instead of one-hot encoding or integer label encoding?
* **Summary Answer**: Entity embeddings map discrete, high-cardinality categorical variables into continuous, low-dimensional vector spaces where similar categories are learned close together. Treating categories as continuous integers falsely imposes Euclidean distance, while one-hot encoding creates a sparse, memory-prohibitive feature matrix ($> 20,000$ columns) that hinders gradient descent.
* **Technical Defense**:
  * In the IEEE-CIS dataset, variables like `card1` (issuer bank) have over $13,500$ unique categories. One-hot encoding these would explode memory and suffer from extreme sparsity.
  * Embeddings represent categories as dense vectors $\mathbf{e}_c \in \mathbb{R}^d$ updated via backpropagation. Through training, issuing banks or email domains with similar fraud behaviors cluster together in latent space.
  * Unseen categories during inference are safely mapped to a dedicated `<UNK>` embedding index (index 1), preventing out-of-vocabulary crashes.

#### Q2: How did you determine the embedding dimensions for each of the 49 categorical features?
* **Summary Answer**: We used a standard empirical heuristic that scales logarithmically with category cardinality:
  $$d_j = \min\left(50, \max\left(4, \text{round}\left(6 \times V_j^{0.25}\right)\right)\right)$$
* **Technical Defense**:
  * Low-cardinality features (e.g., `DeviceType`, $V=4$) receive compact vectors ($d=8$).
  * High-cardinality features (e.g., `card1`, $V=13,554$) are capped at $d=50$ to avoid parameter bloat and overfitting.
  * Summing across all 49 categorical features yields **568 total embedding dimensions**.

#### Q2B: Why did you choose the IEEE-CIS dataset instead of the popular Kaggle European Credit Card Fraud dataset?
* **Summary Answer**: The Kaggle European dataset is composed strictly of 28 anonymized PCA numerical features ($V_1$ to $V_{28}$) and has zero categorical variables. Selecting it would make categorical entity embeddings impossible without fabricating synthetic categories. The IEEE-CIS benchmark contains 49 genuine, high-cardinality categorical entities (card issuers, email domains, device signatures), legitimately justifying the hybrid neural architecture.
* **Technical Defense**:
  * Good ML engineering practice requires using real data that genuinely matches the proposed architecture.
  * In IEEE-CIS, we have 49 real-world categorical variables across product codes (`ProductCD`), payment card attributes (`card1` to `card6`), geographic codes (`addr1`, `addr2`), email domains (`P_emaildomain`, `R_emaildomain`), device platforms (`DeviceType`, `DeviceInfo`), and identity match indicators (`M1` to `M9`, `id_12` to `id_38`).
  * Learning embeddings on these features yielded a $+20.9\%$ relative boost in Test PR-AUC, demonstrating real empirical value.

#### Q3: Why is the input to the first dense layer exactly `Linear(949, 256)`?
* **Summary Answer**: Because the neural network concatenates $381$ continuous numerical features with $568$ total categorical embedding dimensions:
  $$381\text{ (numerical)} + 568\text{ (embeddings)} = 949\text{ total inputs}$$
* **Technical Defense**:
  * The raw transaction has 381 continuous numerical features (amounts, counts, time deltas, V-features) standardized via `StandardScaler`.
  * The 49 categorical variables pass through their respective embedding lookup tables, yielding vectors totaling 568 dimensions.
  * Concatenating these tensors produces an input vector of dimension 949, passed directly into `BatchNorm1d(949)` followed by `Linear(949, 256)`.

---

### Section 2: Imbalance Handling & Loss Formulations

#### Q4: Why is classification accuracy completely unacceptable for evaluating fraud detection?
* **Summary Answer**: In a dataset with $1.91\%$ fraud prevalence, a trivial dummy classifier predicting every transaction as legitimate achieves **$98.09\%$ accuracy** while detecting exactly zero fraud attacks.
* **Technical Defense**:
  * Accuracy places equal weight on False Positives and False Negatives, ignoring the extreme class asymmetry.
  * In fraud detection, the critical metrics are **PR-AUC (Average Precision)**, **Recall** (catching fraud attacks), and **Precision** (avoiding false alarms).

#### Q5: Why did you use `nn.BCEWithLogitsLoss` rather than placing `nn.Sigmoid()` inside the model followed by `nn.BCELoss`?
* **Summary Answer**: Numerical stability. `BCEWithLogitsLoss` combines the sigmoid activation and cross-entropy loss into a single mathematical step using the log-sum-exp trick, preventing arithmetic overflow and underflow.
* **Technical Defense**:
  * If sigmoid outputs an extreme probability ($P = 1.0$ or $0.0$) due to floating-point truncation, $\log(P)$ or $\log(1-P)$ results in $\text{NaN}$ or $-\infty$, causing gradient updates to blow up.
  * `BCEWithLogitsLoss` reformulates the loss:
    $$\mathcal{L}(z, y) = \max(z, 0) - z \cdot y + \log\left(1 + e^{-|z|}\right)$$
    Which is stable across the entire real number line $(-\infty, \infty)$. During inference, sigmoid is applied once to obtain the probability.

#### Q6: How does Weighted BCE address class imbalance, and how is `pos_weight` calculated?
* **Summary Answer**: `pos_weight` multiplies the loss contribution of positive (fraud) instances. It is set to the ratio of negative to positive examples in the training set:
  $$w_{\text{pos}} = \frac{N_{\text{neg}}}{N_{\text{pos}}} = \frac{33,997}{1,003} \approx 28.0$$
* **Technical Defense**:
  * Standard BCE treats each transaction equally, meaning the $97.13\%$ legitimate transactions dominate gradient descent.
  * By scaling positive loss by $28.0$, a missed fraud instance incurs 28 times the gradient penalty of a false alarm, forcing the network to learn rich representations for the minority class.

#### Q7: Why did you experiment with Focal Loss, and what did you observe?
* **Summary Answer**: Focal Loss dynamically down-weights easily classified legitimate examples via a modulating factor $(1 - p_t)^\gamma$, directing optimization toward ambiguous borderline transactions.
* **Technical Defense**:
  * While Weighted BCE scales all positives equally, Focal Loss focuses on *hard* examples.
  * Experimentally, we benchmarked $\gamma \in \{1.0, 2.0, 3.0\}$.
  * **Key Finding**: While Weighted BCE achieved the highest raw PR-AUC ($0.4627$) and Precision ($58.33\%$), **Focal Loss ($\gamma=3$) achieved the best probability calibration**, moving the optimal threshold from an extreme $\theta^* = 0.95$ down to an intuitive mid-range $\theta^* = 0.45$ while generating only $102$ False Positives.

---

### Section 3: Baseline Comparison & Experimental Findings

#### Q8: How did the PyTorch Hybrid model compare to classical tree models like XGBoost and Random Forest?
* **Summary Answer**: XGBoost achieved the highest raw Test PR-AUC ($0.4741$ vs $0.4627$) and Recall ($49.18\%$), but the **PyTorch Hybrid (Weighted BCE)** model achieved superior operational efficiency: lowest False Positives ($95$ vs $126$) and highest Test Precision (**$58.33\%$** vs $54.35\%$).
* **Technical Defense**:
  * In commercial banking, False Positives trigger manual analyst reviews and card re-issuances that cost $\$10$ to $\$15$ each and irritate legitimate cardholders.
  * The PyTorch Hybrid model suppressed False Positives by **$24.6\%$** compared to XGBoost ($95$ vs $126$ FP), providing an optimal risk-reward balance for financial institutions prioritizing alert accuracy.

#### Q9: What did the ablation study prove?
* **Summary Answer**: It proved that adding categorical entity embeddings is responsible for a **$+20.9\%$ relative boost** in Test PR-AUC over the numerical-only neural network.
* **Technical Defense**:
  * Numerical MLP: Test PR-AUC = $0.3828$, Test ROC-AUC = $0.8277$.
  * Hybrid Network (+49 Embeddings): Test PR-AUC = $0.4627$, Test ROC-AUC = $0.8799$.
  * This empirically demonstrates that categorical variables (email domains, hardware device signatures, card networks) contain orthogonal, non-redundant predictive signals that continuous features alone cannot capture.

---

### Section 4: Data Leakage Prevention & Methodology

#### Q10: How did you strictly prevent data leakage throughout the machine learning pipeline?
* **Summary Answer**: We enforced a strict chronological temporal split before fitting any transformations, fitted all imputation and scaling parameters solely on the training split, and selected decision thresholds using validation data only.
* **Technical Defense**:
  * **Temporal Splitting**: Transactions were ordered by `TransactionDT` and split ($70\%$ Train, $15\%$ Val, $15\%$ Test) *before* initializing preprocessors, preventing future transactions from leaking into the past.
  * **Frozen Preprocessing**: Medians and `StandardScaler` ($\mu, \sigma$) were computed strictly on `train_df`. Validation and test data were transformed using these frozen parameters.
  * **Threshold Calibration**: The operating threshold $\theta^* = 0.80$ was tuned strictly on validation set cost curves, preventing test-set data snooping.
  * **Static Audit**: Automated integration tests confirm `src/inference.py` and `app/app.py` contain zero calls to `.fit()` or `.fit_transform()`.

---

### Section 5: Decision Thresholds & Business Cost Utility

#### Q11: Why isn't $\theta = 0.50$ automatically the right threshold for a fraud detection model?
* **Summary Answer**: A threshold of $0.50$ assumes symmetric error costs ($C_{\text{FP}} = C_{\text{FN}}$). In banking, missing a $\$1,000$ fraud attack is far more damaging than conducting a $\$15$ verification review. The threshold is an operational business decision, not a fixed mathematical constant.
* **Technical Defense**:
  * We formulated an explicit banking cost objective:
    $$\text{Total Cost} = \sum_{\text{FN}} \text{TransactionAmt} + C_{\text{FP}} \times |\text{FP}|$$
  * Evaluating across thresholds on validation data, $\theta^* = 0.80$ minimized total financial cost at $C_{\text{FP}} = \$15$, saving **$\$12,140$** on the test set and cutting the manual review queue by $78.9\%$ compared to a naive threshold.

---

### Section 6: Explainability & Deployment Architecture

#### Q12: How does the model explain why a transaction was flagged, and can you claim causality?
* **Summary Answer**: We implemented local sensitivity attribution via counterfactual baseline substitution. We **do not claim causality**; attributions quantify local model sensitivity to specific feature values relative to baseline imputation.
* **Technical Defense**:
  * For each key feature, we replace the observed value with its training median or `<MISSING>` token to measure the drop in predicted probability $\Delta P$. Features producing the largest positive $\Delta P$ are presented as supporting evidence.
  * Because many IEEE-CIS features are anonymized ($V$-features), claiming real-world causal mechanisms would be unscientific and academically dishonest.

#### Q13: Does the Streamlit web dashboard retrain the model or fit scalers?
* **Summary Answer**: Absolutely not. The Streamlit dashboard is purely a presentation layer that calls `FraudInferencePipeline.from_artifacts()`, loading frozen checkpoint weights and preprocessor joblibs.
* **Technical Defense**:
  * The dashboard has zero training routines.
  * Single transaction inference executes in $\sim 140\text{ ms}$.
  * Batch inference processes 500 transactions in under $0.35$ seconds ($> 1,500\text{ txn/sec}$).
