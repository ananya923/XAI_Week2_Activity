# Team Name: ASK

Assumption Junction — interpretable churn models for a telecom company.

**Question:** Which model should the company use if it wants an interpretable churn model, and what assumptions should it worry about before acting on the explanation?

## Contributors

| Student | Model | Notebook |
|---|---|---|
| Ananya Jogalekar | Linear regression (linear probability model) | [`ananya_ASK_linear_regression.ipynb`](ananya_ASK_linear_regression.ipynb) |
| Shelly Cao | Logistic regression | [`ShellyCao_ASK_logistic_regression.ipynb`](ShellyCao_ASK_logistic_regression.ipynb) |
| Kedar Vaidya | Generalized additive model (logistic GAM) | [`Kedar_ASK_gam.ipynb`](Kedar_ASK_gam.ipynb) |

## Dataset

IBM **Telco Customer Churn** (`WA_Fn-UseC_-Telco-Customer-Churn.csv`): 7,043 customers and 21 columns. Each row is one customer at a single snapshot. Predictors cover demographics, phone/internet add-ons, contract and billing, plus `tenure`, `MonthlyCharges`, and `TotalCharges`. The target `Churn` is whether the customer left in the last month (~26.5% yes / ~73.5% no).

The business task is to rank customers by churn risk *and* to explain that ranking in language a retention team can use. Because a dummy that always predicts “stay” is already ~73% accurate, we do not lead with accuracy.

Cleaning shared across notebooks: 11 blank `TotalCharges` values (all `tenure = 0`, none churned) are dropped or coerced; `customerID` is dropped; `Churn` is coded 0/1. `TotalCharges` is nearly tenure × bill, so it is a collinearity/concurvity hazard if kept with both `tenure` and `MonthlyCharges`.

## Assumption Checks

| Model | Key Assumptions Checked | Evidence | Concern |
|---|---|---|---|
| Linear regression | Linearity of \(E[Y \mid X]\); independent errors; residual normality; no severe multicollinearity; no high-leverage outliers | RESET test rejects linearity (\(F = 206\), \(p \approx 10^{-46}\)). Durbin–Watson \(= 2.00\), residual ACF near 0 (independence OK on this snapshot). Shapiro–Wilk \(p \approx 10^{-33}\), Jarque–Bera \(p \approx 10^{-84}\) (normality fails). VIF: `MonthlyCharges` \(= 866\), fiber dummy \(= 149\), `TotalCharges` \(= 11\). Cook’s \(D\): 0 points \(> 1\) | Binary \(Y\) is the wrong family for OLS. Predicted “probabilities” can fall outside \([0, 1]\). Collinear bill/service terms make individual coefficients untrustworthy (including a counterintuitive negative `MonthlyCharges` sign). |
| Logistic regression | Bernoulli / logit; independence; linearity in the *log-odds*; no severe multicollinearity; usable class balance | Binary outcome at ~27% churn; one row per customer ID. `tenure`–`TotalCharges` \(r = 0.83\). Test ROC-AUC \(= 0.832\). Default threshold 0.5: recall \(= 0.516\) on the churn class | Log-odds linearity is assumed, not shown. The GAM later finds a tenure cliff then a flat region, so a single logistic slope for tenure is a simplification. Keeping `TotalCharges` with tenure/bill splits credit across collinear terms. The 0.5 cutoff under-detects churners. |
| GAM | Bernoulli / logit; independence; *smooth* (not linear) continuous effects; additivity; low concurvity | Logit is the right family. 7,032 unique IDs, one row each. Empirical log-odds of tenure fall then flatten; monthly charges are not a monotone line. Spline \(R^2\) of `TotalCharges` on tenure \(= 0.69\) (dropped from the GAM). Contract × internet: fiber churn rate \(0.55\) on month-to-month vs \(0.07\) on two-year; tenure curves are not parallel shifts | Additivity is the assumption most at risk. One global tenure plot averages month-to-month (steep) and two-year (flat). Tensor terms did not beat additive AIC, but they *changed the briefing* for a new two-year customer (~46% vs ~16%). pygam \(p\)-values after \(\lambda\) search are not real tests. |

## Model Comparison

| Model | Performance Evidence | Interpretability Strength | Interpretability Weakness |
|---|---|---|---|
| Linear regression | Test \(R^2 = 0.252\), MSE \(= 0.146\) (RMSE \(\approx 0.38\)). In-sample \(R^2 = 0.284\). Ridge matches OLS; Lasso collapses most coefficients and \(R^2\) falls to \(0.123\) | Coefficients read as \(\Delta P(\text{churn})\) in probability units, which is easy to brief | Wrong model for a 0/1 label. Unbounded predictions. Collinearity (fiber vs monthly bill vs total charges) can flip signs, so the “story” is not stable enough to act on. |
| Logistic regression | Accuracy \(0.788\), precision \(0.623\), recall \(0.516\), F1 \(0.564\), ROC-AUC \(0.832\) | Odds ratios with a holding-other-variables-fixed reading. Two-year contract OR \(\approx 0.27\) vs month-to-month; fiber OR \(\approx 1.94\). Standard, communicable briefing for a retention team | One slope per feature: cannot show that churn risk is front-loaded in tenure. Odds are not probabilities. Coefficients are associations, not causal effects of changing a plan. |
| GAM | Test ROC-AUC \(0.840\), PR-AUC \(0.654\), Brier \(0.138\), accuracy \(0.797\). 5-fold CV AUC \(0.849 \pm 0.005\). Same-features logistic: AUC \(0.834\) (a tie). Well calibrated by risk decile | Each term is one plot: tenure contribution \(+1.79\) at month 1, ~0 near month 22, \(-1.30\) at month 72. Contract still dominates among categoricals. Captures the nonlinear shapes logistic has to ignore | Extra flexibility buys *shape*, not ranking. Additive PDPs can mis-brief segments (new two-year customers). Curves are easy to over-read; smoothing penalty \(\lambda\) is a modeling choice. Still not causal. |

## Recommendation

**Recommended model:** logistic regression as the company-facing interpretable model, with the GAM kept as a diagnostic (especially for tenure and monthly charges). Do not use the linear probability model for this churn task.

**Why this model:** The company asked for an *interpretable* churn model it can brief and act on. Logistic regression and the GAM rank customers almost the same (AUC \(0.832\)–\(0.840\)). The logistic odds ratios are the most operational explanation: “two-year vs month-to-month,” “fiber vs not,” holding other recorded features fixed. The GAM shows *why* a linear-in-log-odds slope for tenure is a simplification (risk is front-loaded), but that extra description does not improve ranking enough to justify shipping a harder-to-audit smoother as the production explainer. Linear regression fails the outcome family, can predict outside \([0, 1]\), and produces collinear coefficients the company should not quote.

**What the company can responsibly conclude:**

- Month-to-month contracts, fiber internet, electronic check, and missing security/tech-support add-ons are *associated* with higher churn odds; longer contracts and those add-ons are associated with lower odds.
- New customers (low tenure) are the highest-risk window; a “predict stay” policy already looks accurate because most customers stay.
- GAM scores on this snapshot are usable as probabilities for targeting lists (calibration hugs the diagonal), but the default 0.5 cutoff misses about half of actual churners.

**What the company should not conclude yet:**

- Moving a customer onto a two-year contract *causes* them to stay. Contract type also marks customers who already intended to stay.
- Fiber *causes* churn, or that raising/lowering the monthly bill has a simple linear effect. The linear model’s negative `MonthlyCharges` coefficient is a collinearity artifact, not a reason to raise prices.
- Accuracy near 80% means the model is catching people who will leave. Recall at 0.5 is ~0.52.
- An additive GAM tenure plot describes every contract the same way. A new two-year customer is not a new month-to-month customer with a constant shift.

**One next analysis we would run:** estimate a tenure × contract term (or a monotone tenure spline) and, separately, a survival / time-to-churn model. The current labels are a last-month snapshot, not a duration process, and that is the assumption that most limits any of these explanations as a policy lever.
