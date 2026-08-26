# Quantitative Financial Risk Engine: Value at Risk (VaR) Analysis under Parametric and Empirical Distributions

**Course:** DSC003 (Part A) — Probability and Statistics  
**Assessment:** Continuous Assessment | Group Project Activity  

---

## 1. Title
**Quantitative Financial Risk Engine: Value at Risk (VaR) Analysis under Parametric and Empirical Distributions**

---

## 2. Background, Motivation, and Rationale
In financial markets, estimating the potential loss of a portfolio is critical for capital adequacy, regulatory compliance (such as Basel III/IV), and general risk management. The standard metric used for this purpose is **Value at Risk (VaR)**. VaR answers the question: *"What is the maximum loss that we could expect to suffer over a given time horizon with a specified level of confidence?"*

The core challenge in calculating VaR is accurately modeling the distribution of financial returns. Historically, financial models assumed that stock returns follow a Normal (Gaussian) distribution due to the Central Limit Theorem. However, real-world asset returns exhibit "leptokurtosis" (fat tails and high peaks), meaning extreme market crashes (such as the 2008 Financial Crisis or the 2020 COVID-19 crash) occur far more frequently than a Normal distribution predicts. Underestimating these tails leads to severe under-capitalization and risk exposure. 

This project builds a quantitative risk engine to analyze daily returns of the **S&P 500 Index** over a 10-year period (2016–2026). We analyze and compare the empirical distribution of returns against parametric models fitted using both the Normal and Student's t-distributions, utilizing the Cumulative Distribution Function (CDF) as our primary analytical framework.

---

## 3. Statistical Theory and Concept (Main Spotlight: CDF)
The mathematical foundation of this project is built upon the **Cumulative Distribution Function (CDF)**, the **Quantile Function**, and the convergence properties that bridge empirical observations with theoretical probability models.

### 3.1. The Continuous Cumulative Distribution Function (CDF)
Let $X$ represent the continuous random variable of daily asset returns. The Cumulative Distribution Function (CDF), denoted by $F(x)$, is defined as the probability that $X$ takes a value less than or equal to $x$:
$$F(x) = \Pr(X \le x) = \int_{-\infty}^{x} f(t) \, dt$$
where $f(t)$ is the probability density function (PDF). 

The CDF is mathematically constrained by the following fundamental properties (DeGroot & Schervish, Section 3.3):
1.  **Bounds:** $0 \le F(x) \le 1$ for all $x \in \mathbb{R}$.
2.  **Monotonicity:** $F(x)$ is non-decreasing; if $x_1 < x_2$, then $F(x_1) \le F(x_2)$.
3.  **Limits:** $\lim_{x \to -\infty} F(x) = 0$ and $\lim_{x \to \infty} F(x) = 1$.
4.  **Right-Continuity:** $F(x) = \lim_{y \to x^+} F(y)$.

For a continuous distribution, $F(x)$ is continuous everywhere, and by the Fundamental Theorem of Calculus:
$$\frac{d}{dx}F(x) = f(x)$$

### 3.2. The Quantile Function and Value at Risk (VaR)
To determine the threshold of downside risk, we must invert the CDF. For any probability level $p \in (0, 1)$, the **Quantile Function** $F^{-1}(p)$ is defined as (DeGroot & Schervish, Definition 3.3.2):
$$F^{-1}(p) = \inf \{ x \in \mathbb{R} : F(x) \ge p \}$$

In risk management, let $Y = -X$ represent the daily loss (negative return). If we specify a confidence level $c \in (0, 1)$ (typically $c = 0.95$ or $c = 0.99$), the **Value at Risk (VaR)** is the $c$-quantile of the loss distribution:
$$\text{VaR}_c = F_Y^{-1}(c)$$

Alternatively, mapping this back to the return distribution $X$, the return threshold $x_0$ corresponding to a probability of exceeding loss is the $(1-c)$-quantile:
$$\Pr(X \le x_0) = 1-c \implies x_0 = F_X^{-1}(1-c)$$
Because daily losses are expressed as positive percentages, VaR is the negative of this return quantile:
$$\text{VaR}_c = -x_0 = -F_X^{-1}(1-c)$$
*   For **95% VaR**: $\text{VaR}_{0.95} = -F_X^{-1}(0.05)$
*   For **99% VaR**: $\text{VaR}_{0.99} = -F_X^{-1}(0.01)$

### 3.3. The Empirical CDF (eCDF)
To compute VaR directly from historical data (known as Historical Simulation), we construct the **Empirical CDF**, $F_n(x)$. For a sample of $n$ independent and identically distributed (i.i.d.) observations $X_1, X_2, \dots, X_n$ (sorted in ascending order), the eCDF is defined as a step function:
$$F_n(x) = \frac{1}{n} \sum_{i=1}^{n} I(X_i \le x)$$
where $I(\cdot)$ is the indicator function:
$$I(X_i \le x) = \begin{cases} 1 & \text{if } X_i \le x \\ 0 & \text{if } X_i > x \end{cases}$$
The eCDF represents the proportion of historical trading days with returns less than or equal to $x$.

### 3.4. The Glivenko-Cantelli Convergence
The mathematical justification for using historical returns to estimate future risk lies in the **Glivenko-Cantelli Theorem** (frequently referred to as the Fundamental Theorem of Statistics). 

Let $X_1, X_2, \dots, X_n$ be i.i.d. observations with a true underlying CDF $F(x)$. The theorem states that the empirical CDF $F_n(x)$ converges uniformly to the true CDF $F(x)$ almost surely (with probability 1) as the sample size $n$ approaches infinity:
$$\lim_{n \to \infty} \sup_{x \in \mathbb{R}} |F_n(x) - F(x)| = 0 \quad \text{a.s.}$$

This means that as our sample size of daily returns grows larger, the empirical risk step function becomes an increasingly accurate and consistent estimator of the true underlying probability distribution of returns.

### 3.5. Parametric Fitting (Normal vs. Student's t)
Rather than relying solely on historical returns, we can assume a parametric form for the CDF and estimate its parameters using **Maximum Likelihood Estimation (MLE)**.

#### 3.5.1. Normal Distribution Fit
If returns follow a Normal distribution, $X \sim \mathcal{N}(\mu, \sigma^2)$, the theoretical CDF is given by:
$$F_{\text{Normal}}(x) = \Phi\left(\frac{x - \mu}{\sigma}\right) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{\frac{x - \mu}{\sigma}} e^{-\frac{t^2}{2}} \, dt$$
Using MLE, we estimate the mean $\hat{\mu} = \frac{1}{n}\sum X_i$ and standard deviation $\hat{\sigma} = \sqrt{\frac{1}{n}\sum (X_i - \hat{\mu})^2}$. The parametric Normal VaR is then:
$$\text{VaR}_c = -(\hat{\mu} + Z_{1-c} \hat{\sigma})$$
where $Z_{1-c}$ is the standard normal quantile (e.g., $Z_{0.05} \approx -1.645$, $Z_{0.01} \approx -2.326$).

#### 3.5.2. Student's t-Distribution Fit
To capture heavy tails, we fit a Student's t-distribution with location parameter $\mu$, scale parameter $s$, and degrees of freedom $\nu$. The PDF is:
$$f_t(x; \nu, \mu, s) = \frac{\Gamma\left(\frac{\nu+1}{2}\right)}{\Gamma\left(\frac{\nu}{2}\right)\sqrt{\pi \nu} s} \left[1 + \frac{1}{\nu}\left(\frac{x - \mu}{s}\right)^2\right]^{-\frac{\nu+1}{2}}$$
The CDF $F_t(x)$ is calculated by integrating this density numerically. The parameter $\nu$ controls the thickness of the tails: lower values of $\nu$ indicate heavier tails. As $\nu \to \infty$, the Student's t-distribution converges to the Normal distribution. Parametric Student's t VaR is:
$$\text{VaR}_c = -(\hat{\mu} + t_{1-c, \hat{\nu}} \hat{s})$$
where $t_{1-c, \hat{\nu}}$ is the quantile of the standard Student's t-distribution with $\hat{\nu}$ degrees of freedom.

### 3.6. Kolmogorov-Smirnov (KS) Goodness-of-Fit Test
To mathematically determine which distribution models the returns more accurately, we perform a **Kolmogorov-Smirnov Goodness-of-Fit Test**. The test compares the Empirical CDF $F_n(x)$ against a fitted parametric CDF $F_{\text{model}}(x)$. 

The test statistic $D_n$ represents the supremum (maximum vertical distance) between the two curves:
$$D_n = \sup_{x \in \mathbb{R}} |F_n(x) - F_{\text{model}}(x)|$$

We test the hypotheses:
*   $H_0$: The returns follow the fitted distribution ($F(x) = F_{\text{model}}(x)$).
*   $H_1$: The returns do not follow the fitted distribution ($F(x) \neq F_{\text{model}}(x)$).

A smaller $D_n$ and a larger p-value indicate a better fit. If the p-value is less than our significance level (e.g., $\alpha = 0.05$), we reject the null hypothesis, concluding that the distribution is not a suitable model for the data.

---

## 4. Real-Life Example/Case Study
We implement this risk engine using daily historical close prices of the **S&P 500 Index** (ticker: `^GSPC`) downloaded from Yahoo Finance.
*   **Sample Period:** January 1, 2016 to January 1, 2026
*   **Total Sample Size ($n$):** 2,513 daily return observations
*   **Asset Class:** Large-cap US Equities index (representing real-world diversified stock portfolio risk).

For each trading day $t$, we calculate the logarithmic return from the adjusted closing price $P_t$:
$$R_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$$
Logarithmic returns are preferred over simple returns because they are time-additive and track continuous compounding.

---

## 5. Results
The quantitative risk engine generated the following results from the S&P 500 return dataset:

### 5.1. Maximum Likelihood Parameter Estimates
*   **Fitted Normal Distribution Parameters:**
    *   $\hat{\mu} = 0.000487$ (representing a daily mean return of $\approx 0.049\%$)
    *   $\hat{\sigma} = 0.011449$ (representing daily volatility of $\approx 1.145\%$)
*   **Fitted Student's t-Distribution Parameters:**
    *   Degrees of Freedom ($\hat{\nu}$): $2.5389$ (very low value, confirming extremely fat tails)
    *   Location ($\hat{\mu}$): $0.000951$
    *   Scale ($\hat{s}$): $0.006250$

### 5.2. Value at Risk (VaR) Comparisons
The table below displays the daily VaR estimates (expressed as positive loss percentages) calculated using the three methods:

| Confidence Level ($c$) | Empirical VaR (Historical) | Parametric Normal VaR | Parametric Student's t VaR |
| :--- | :---: | :---: | :---: |
| **95.0% Confidence** | $1.715\%$ | $1.834\%$ | $1.491\%$ |
| **99.0% Confidence** | $3.401\%$ | $2.615\%$ | $3.199\%$ |

### 5.3. Kolmogorov-Smirnov Goodness-of-Fit Test Results
*   **Normal Distribution Fit:**
    *   KS Statistic ($D_n$): $0.11322$
    *   p-value: $1.62 \times 10^{-28}$
*   **Student's t-Distribution Fit:**
    *   KS Statistic ($D_n$): $0.02462$
    *   p-value: $0.0935$ ($9.35\%$)

### 5.4. Analysis of the Comparative CDF Plot (Tail Zoom)
The generated plot (`var_cdf_comparison.png`) illustrates the CDFs. 
*   **Main Plot:** The three curves appear to overlap closely, representing the body of the distribution.
*   **Left-Tail Zoom (Inset):** The inset zooms into the critical loss region (daily returns from $-6\%$ to $-0.5\%$, where probabilities are under $8\%$). Here, the structural failure of the Normal CDF model is laid bare:
    *   At the 99% confidence level, the **Empirical VaR** is **$3.401\%$**. 
    *   The **Normal CDF** yields a VaR of only **$2.615\%$**. This represents a severe underestimation of risk. If a bank managed its capital using the Normal model, a 99% risk limit would be breached far more often than 1% of the time, leading to unexpected insolvency.
    *   The **Student's t CDF** yields a VaR of **$3.199\%$**, which matches the empirical tail much closer. It captures the slow decay of the probability tail.

The KS test formally validates this. The p-value for the Normal distribution ($1.62 \times 10^{-28}$) is effectively zero, meaning we reject the hypothesis that S&P 500 returns are normally distributed. In contrast, the Student's t-distribution yields a p-value of $9.35\%$. Since $0.0935 > 0.05$, we fail to reject the null hypothesis at the 5% significance level, proving that the Student's t-distribution is a statistically compatible fit for S&P 500 returns.

---

## 6. Conclusion and Limitations

### 6.1. Key Takeaways
1.  **CDF is the Core Risk Framework:** The Cumulative Distribution Function is the natural framework for evaluating risk because it maps return thresholds directly to cumulative probabilities of losses.
2.  **The Normal Distribution Underestimates Tail Risk:** The Gaussian assumption fails to model financial assets. At 99% confidence, the Normal model underestimates the S&P 500 daily loss threshold by approximately $79$ basis points ($2.615\%$ vs $3.401\%$).
3.  **Student's t-Distribution is Superior:** By incorporating the degrees of freedom parameter ($\nu \approx 2.54$), the Student's t-distribution captures leptokurtosis, matches the empirical CDF tail, and passes the Kolmogorov-Smirnov test.

### 6.2. Limitations of VaR
While VaR is a valuable risk metric, it has significant statistical limitations:
1.  **No Information Beyond Threshold:** VaR only states the threshold of loss (e.g., $3.40\%$ loss at 99% confidence). It provides no information about the magnitude of losses *beyond* this threshold (the shape of the tail past 99%).
2.  **Non-Subadditivity:** VaR is not a mathematically "coherent" risk measure because it violates subadditivity. The VaR of a combined portfolio can sometimes be greater than the sum of the individual VaRs of its components, discouraging diversification.
3.  **Stationarity Assumption:** Parametric and historical VaR models assume that historical volatility and distribution shapes remain constant over time (stationarity), which fails during sudden market structural shifts.

To address these limitations, modern financial systems augment VaR with **Expected Shortfall (ES)** (also called Conditional VaR), which integrates the tail of the CDF beyond the VaR quantile to calculate the average loss in the worst-case scenario.

---

## 7. References
1.  DeGroot, M. H., & Schervish, M. J. (2012). *Probability and Statistics* (4th Edition). Pearson.
    *   *Section 3.3: The Cumulative Distribution Function (pages 107–116).*
    *   *Section 3.4: Bivariate Distributions & Quantile/VaR Examples (pages 118–129).*
2.  Hull, J. C. (2018). *Risk Management and Financial Institutions* (5th Edition). Wiley.
3.  Glivenko, V. (1933). Sulla determinazione empirica della legge di probabilità. *Giornale dell'Istituto Italiano degli Attuari*, 4, 92–99.
