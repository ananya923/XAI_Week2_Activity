# Team Name: ASK

**Assumption Junction** — interpretable churn models for a telecom company.

**Guiding question:** Which model should the company use if it wants an interpretable churn model, and what assumptions should it worry about before acting on the explanation?

## Contributors

| Student | Model | Notebook |
|---|---|---|
| Ananya Jogalekar | Linear regression | [`ananya_ASK_linear_regression.ipynb`](ananya_ASK_linear_regression.ipynb) |
| Shelly Cao | Logistic regression | [`ShellyCao_ASK_logistic_regression.ipynb`](ShellyCao_ASK_logistic_regression.ipynb) |
| Kedar Vaidya | GAM | [`Kedar_ASK_gam.ipynb`](Kedar_ASK_gam.ipynb) |

## Dataset

IBM **Telco Customer Churn** (`WA_Fn-UseC_-Telco-Customer-Churn.csv`): 7,043 customers, 21 columns, one row per customer at a single snapshot. Predictors cover demographics, phone and internet add-ons, contract and billing, plus `tenure`, `MonthlyCharges`, and `TotalCharges`. The target `Churn` records whether the customer left in the last month, at roughly 27% yes and 73% no.

The task is to rank customers by churn risk *and* explain that ranking in language a retention team can act on. Because always predicting "stay" is already about 73% accurate, we report ROC-AUC and recall rather than leading with accuracy.

Cleaning is shared across the three notebooks: the 11 blank `TotalCharges` values (all `tenure = 0`, none churned) are dropped, `customerID` is dropped, and `Churn` is coded 0/1. `TotalCharges` is close to tenure times monthly bill, so keeping all three together creates a collinearity and concurvity hazard.

## Assumption Checks

| Model | Key Assumptions Checked | Evidence | Concern |
|---|---|---|---|
| Linear regression | Linearity, independent errors, normal residuals, no severe multicollinearity, no influential outliers | RESET F = 206 (p ≈ 1e-46); Durbin–Watson = 2.00 with residual ACF near zero; Shapiro–Wilk p ≈ 1e-33 and Jarque–Bera p ≈ 1e-84; VIF = 866 for `MonthlyCharges` and 149 for the fiber dummy; no Cook's D above 1 | Linearity and normality both fail. A 0/1 target can produce fitted "probabilities" outside [0, 1], and the collinear billing terms make individual coefficients unstable, including a negative `MonthlyCharges` sign that reverses the expected story. |
| Logistic regression | Bernoulli/logit family, independence, linearity in the log-odds, no severe multicollinearity, usable class balance | Binary outcome at 27% churn, one row per customer ID; `tenure`–`TotalCharges` correlation r = 0.83; test ROC-AUC = 0.832; recall = 0.516 at the default 0.5 threshold | Log-odds linearity is assumed rather than tested. The GAM finds that tenure risk drops sharply then flattens, so a single slope understates early-tenure risk. The 0.5 cutoff misses about half of actual churners. |
| GAM | Bernoulli/logit family, independence, smooth (not linear) effects, additivity, low concurvity | 7,032 unique IDs, one row each; binned log-odds of tenure fall then flatten and `MonthlyCharges` is not monotone; spline R² of `TotalCharges` on tenure = 0.69, so it was dropped; fiber churn is 0.55 on month-to-month versus 0.07 on two-year | Additivity is the weakest link. One global tenure curve averages a steep month-to-month decline with a flat two-year line. Tensor terms did not beat additive AIC but did change the briefing. pygam p-values computed after the smoothing search are not valid tests. |

## Model Comparison

| Model | Performance Evidence | Interpretability Strength | Interpretability Weakness |
|---|---|---|---|
| Linear regression | Test R² = 0.252, MSE = 0.146 (RMSE ≈ 0.38); in-sample R² = 0.284; Ridge matches OLS, Lasso falls to R² = 0.123 | Coefficients are already in probability units, so effects read directly as changes in P(churn) | Wrong outcome family for a 0/1 label, predictions are unbounded, and collinearity flips signs so the story is not stable enough to act on |
| Logistic regression | Accuracy 0.788, precision 0.623, recall 0.516, F1 0.564, ROC-AUC 0.832 | Odds ratios with a hold-everything-else-fixed reading: two-year contract 0.27, fiber 1.94, online security 0.63 | One slope per feature, so it cannot show that risk is front-loaded in tenure; odds are not probabilities |
| GAM | Test ROC-AUC 0.840, PR-AUC 0.654, Brier 0.138, accuracy 0.797; 5-fold CV AUC 0.849 ± 0.005; matched logistic baseline 0.834 | Each effect is one readable curve (tenure contributes +1.79 at month 1, about 0 at month 22, −1.30 at month 72) and scores are well calibrated across deciles | The extra flexibility buys shape, not ranking; additive curves mis-brief segments; the smoothing penalty is a modeling choice that is easy to over-read |

## Recommendation

### Recommended model

Logistic regression as the company-facing interpretable model, with the GAM retained as a diagnostic for tenure and monthly charges. The linear probability model should not be used for this task.

### Why this model

Logistic regression and the GAM rank customers almost identically (ROC-AUC 0.832 versus 0.840, within the spread of the GAM's own cross-validation), so the choice comes down to how well each explains itself. Odds ratios give the retention team a directly quotable statement — two-year versus month-to-month, fiber versus not — while the GAM's smooth curves require more care to read and its smoothing penalty is an extra modeling choice to defend. The GAM still earns its place as a diagnostic: it is what shows that tenure risk is front-loaded rather than linear, which is a fact the logistic model hides. Linear regression is ruled out on the outcome family alone, before its collinearity problems are even considered.

### What the company can responsibly conclude

- Month-to-month contracts, fiber internet, electronic check payment, and missing security or tech-support add-ons are *associated* with higher churn odds; longer contracts and those add-ons are associated with lower odds.
- Low-tenure customers are the highest-risk window, and the risk concentrates in roughly the first year rather than declining steadily.
- GAM scores are calibrated well enough on this snapshot to be used as probabilities for a targeting list, not just as a ranking.

### What the company should not conclude yet

- That moving a customer onto a two-year contract *causes* them to stay. Contract type also marks customers who already intended to stay.
- That fiber causes churn, or that the monthly bill has a simple linear effect. The linear model's negative `MonthlyCharges` coefficient is a collinearity artifact, not a reason to change prices.
- That roughly 80% accuracy means the model catches churners. Recall at the default threshold is about 0.52, so nearly half of actual churners are missed.
- That the additive tenure curve applies to every contract. A new two-year customer is not a new month-to-month customer plus a constant shift; the additive model puts them near 46% where the interaction model puts them near 16%.

### One next analysis we would run

Fit a tenure-by-contract interaction (or a monotone tenure spline) and, separately, a survival model for time to churn. The current labels are a last-month snapshot rather than a duration, and that framing is what most limits any of these explanations as a basis for policy.

## Repository Contents

| File | Description |
|---|---|
| `README.md` | This team report |
| `ananya_ASK_linear_regression.ipynb` | Linear regression: EDA, assumption tests, linear probability model |
| `ShellyCao_ASK_logistic_regression.ipynb` | Logistic regression: EDA, assumption checks, odds-ratio interpretation |
| `Kedar_ASK_gam.ipynb` | Logistic GAM: assumption EDA, smooth effects, interaction tests |
| `WA_Fn-UseC_-Telco-Customer-Churn.csv` | IBM Telco Customer Churn dataset |
| `reference/` | Course starter notebooks |

The GAM notebook requires `pygam` (`pip install pygam`); the other notebooks run on the standard pandas, statsmodels, scikit-learn, matplotlib, and seaborn stack.
