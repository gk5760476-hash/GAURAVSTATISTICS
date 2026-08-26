# Resume Bullet Points & CV Descriptions (VaR Risk Engine Project)

To highlight this project on your CV/resume, here are three pre-written, professional descriptions tailored for different career tracks. Select and adapt the one that matches your target role.

---

### Track 1: Quantitative Finance / Risk Management
> **Quantitative Risk Analyst / FinTech Engineer Project**
> *   Designed and implemented a mathematical **Quantitative Risk Engine** in Python utilizing `yfinance` to evaluate downside volatility and Value at Risk (VaR) for the S&P 500 Index over a 10-year period (2500+ daily observations).
> *   Formulated continuous **Cumulative Distribution Functions (CDFs)** and **Quantile Functions** to compute empirical vs. parametric VaR at 95% and 99% confidence levels.
> *   Conducted Maximum Likelihood Estimation (MLE) fitting of **Normal** and **Student's t-distributions**, demonstrating the risk-underestimation of Gaussian models in high-volatility regimes due to heavy tails.
> *   Validated distribution fits using **Kolmogorov-Smirnov (KS) Goodness-of-Fit Tests**, achieving a Student's t fit passing p-value of 0.0935 vs. Normal fit p-value of $1.6 \times 10^{-28}$, and generated publication-quality comparative CDF plots with inset tail zoom visualization.

---

### Track 2: Data Science / Statistical Modeling
> **Data Scientist / Statistical Modeler Project**
> *   Engineered an end-to-end statistical modeling pipeline in Python to analyze and compare empirical returns of major equity indices against theoretical probability distributions.
> *   Applied the **Glivenko-Cantelli Theorem** to mathematically prove the uniform convergence of the Empirical CDF (eCDF) to the underlying risk distribution as sample size $n \to \infty$.
> *   Utilized **SciPy** and **Pandas** to perform maximum likelihood estimation of multi-parameter continuous probability distributions (Gaussian and heavy-tailed Student's t).
> *   Implemented non-parametric statistical hypothesis testing (**Kolmogorov-Smirnov Test**) to evaluate model residuals and goodness-of-fit, proving the necessity of heavy-tailed models for non-Gaussian datasets.

---

### Track 3: SRE / SaaS Monitoring / Performance SRE
> **Site Reliability Engineer (SRE) / Systems Risk Project**
> *   Developed a Python-based risk analytics engine using the principles of financial **Value at Risk (VaR)** to model and predict extreme system performance outliers and tail latencies.
> *   Modeled high-percentile system behaviors (95th and 99th quantile thresholds) using empirical and parametric continuous CDFs to analyze the frequency and scale of extreme latency crashes.
> *   Utilized **Student's t-distribution fits** to capture heavy-tail server spikes (where degrees of freedom $\nu \approx 2.54$), showing the inaccuracy of standard bell-curve approximations for system metric outliers.
> *   Generated comparative CDF visualization plots with inset tail zoom panels to isolate, analyze, and present performance risk profiles under different traffic patterns.
