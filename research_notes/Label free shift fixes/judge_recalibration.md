# Label-free trust in one stacker feature when the veto rate and the group prior both change

Scope: a gradient-boosted stacker whose inputs include a stage-2 match score \(p\), a pairwise language-model judge (yes / veto), and other competition features. Labeled data exist for source groups (US / India, including leave-one-country-out). The target group (France) has scores and judge outputs but no match labels. The question is how \(P(Y{=}1 \mid p, J, g)\) should move when the marginal veto rate changes, and which published methods can even be fit in that regime.

Notation used throughout:
- \(Y \in \{0,1\}\): true match.
- \(J \in \{0,1\}\): judge says match (\(1\)) or veto (\(0\)).
- \(p\): stage-2 score. \(g\): group. \(s\): labeled source.
- False-veto rate \(f_g(p) = P(J{=}0 \mid Y{=}1, p, g)\).
- True-veto rate \(t_g(p) = P(J{=}0 \mid Y{=}0, p, g)\).
- Bin prior \(\pi_g(p) = P(Y{=}1 \mid p, g)\). Decoy density in the bin \(\delta_g(p) = 1 - \pi_g(p)\).
- Marginal veto rate \(r_g(p) = P(J{=}0 \mid p, g)\).

Law of total probability (no extra assumption), inside a bin of the features one conditions on:

\[
r_g(p) = f_g(p)\,\pi_g(p) + t_g(p)\,(1-\pi_g(p)).
\]

Three group-level unknowns \((f_g, t_g, \pi_g)\), one unlabeled observable \(r_g\). A higher veto rate alone does not identify which unknown moved.

Task-brief rates below are **not published measurements**. They are scenario inputs. No sample size \(n\) was given, so none of the rates has a computed standard error here.

## What is the correct update of P(match | p, judge, group) when the veto rate changes and the group has no labels?

### Takeaway

Under a constant judge likelihood, a higher veto rate means the match prior in that slice fell, so the same Bayes factor \(f/t\) is applied to worse prior odds and a veto should move \(P(\text{match})\) further down. It does **not** mean the veto coefficient itself should be scaled up. If the veto rate exceeds what the source false-veto rate, the source true-veto rate, and an anchor prior from \(p\) can produce, that constant-likelihood update is impossible and using it trusts a broken judge too much. Saerens EM and BBSE implement only the constant-likelihood (label-shift) case. Platt and temperature scaling need labels. Importance weighting never revises a coefficient because reliability changed.

### Cited Findings

- Absent an assumption that source and target share something, correcting a classifier with no target labels is impossible. Label shift is \(q(y,x) = q(y)\,p(x \mid y)\). Covariate shift is \(q(y,x) = q(x)\,p(y \mid x)\). — [Lipton, Wang, Smola, ICML 2018](https://ar5iv.labs.arxiv.org/html/1802.03916)
- When the new class priors are known, the source posterior is reweighted by the prior ratio. Saerens, Latinne, and Decaestecker’s EM does this when the priors are unknown: initialize \(\hat q^{(0)}(y)\) at the source prior, then
  \[
  \hat q^{(s)}(y{=}i \mid x_k)
  = \frac{\frac{\hat q^{(s)}(y{=}i)}{\hat p(y{=}i)}\,\hat p(y{=}i \mid x_k)}{\sum_j \frac{\hat q^{(s)}(y{=}j)}{\hat p(y{=}j)}\,\hat p(y{=}j \mid x_k)},
  \qquad
  \hat q^{(s+1)}(y{=}i) = \frac{1}{N}\sum_k \hat q^{(s)}(y{=}i \mid x_k).
  \]
  The E-step is exactly the prior-ratio correction. The M-step sets the new prior to the average corrected posterior on unlabeled target rows. They present this as EM for the target likelihood, with no model refit. — [Alexandari, Kundaje, Shrikumar, ICML 2020, quoting Saerens et al. 2002](https://ar5iv.labs.arxiv.org/html/1901.06852); [Saerens, Latinne, Decaestecker, Neural Computation 2002](https://pubmed.ncbi.nlm.nih.gov/11747533/)
- Assumptions stated for that EM: \(p(x \mid y)\) is unchanged, and \(\hat p(y \mid x)\) is the calibrated source posterior. Alexandari et al. prove the label-shift maximum-likelihood objective is concave, so EM reaches the global maximum when those assumptions hold. Miscalibration of modern networks breaks the calibration assumption. Temperature scaling reduced ECE in Guo et al. but left systematic bias (class-average predictions drifting from class frequencies) that hurt EM. Variants with class-specific bias terms (vector scaling; bias-corrected temperature scaling) did better as a pre-step. Vector scaling is \(\mathrm{softmax}(W_i z_i + b_i)\). — [Alexandari et al. 2020](https://ar5iv.labs.arxiv.org/html/1901.06852)
- Temperature scaling itself is \(p(y{=}i \mid x) = \exp(z_i / T) / \sum_j \exp(z_j / T)\), with \(T\) fit by negative log-likelihood on a **labeled** held-out source set. It is a one-parameter restriction of Platt scaling. — [Alexandari et al. 2020, attributing the formula to Guo et al. 2017](https://ar5iv.labs.arxiv.org/html/1901.06852)
- Platt scaling is the two-parameter map \(P(y{=}1 \mid x) = 1 / (1 + \exp(A f(x) + B))\), with \(A,B\) fit by maximum likelihood on labels (held-out or cross-validated; Platt also shrinks the binary targets). A nonzero \(B\) shifts the probability threshold relative to \(\mathrm{sign}(f)\). — [Platt scaling, Wikipedia](https://en.wikipedia.org/wiki/Platt_scaling); [LibSVM / sklearn note](https://stackoverflow.com/questions/15111408/how-does-sklearn-svm-svcs-function-predict-proba-work-internally)
- Black-box shift estimation does **not** need calibrated probabilities. Estimate weights \(w(y) = q(y)/p(y)\) by a linear system: the source confusion matrix of a fixed predictor times \(w\) equals the average predictor output on unlabeled target data. Requirements: label shift (so \(q(\hat y \mid y) = p(\hat y \mid y)\)), every target class seen in the source, and an invertible confusion matrix. Consistency and a test of \(w = 1\) are proved under those assumptions. The correction Lipton et al. then apply is importance-weighted empirical risk minimization on source labels, not a free change of one coefficient. — [Lipton, Wang, Smola 2018](https://ar5iv.labs.arxiv.org/html/1802.03916)
- Under covariate shift, the fix is the opposite factorization: reweight the training loss by \(w(x) = q(x)/p(x)\). Sugiyama’s review writes importance-weighted logistic regression as minimizing \(\sum_i w(x_i)^\gamma \log(1 + \exp(-y_i f_\theta(x_i))) + \lambda \|\theta\|^2\), citing Shimodaira 2000. The ratio is a property of the marginal of \(x\), estimable without target labels by classifying source versus target covariates, \(\hat w(x) = \hat P(C{=}1 \mid x) / (1 - \hat P(C{=}1 \mid x))\). — [Sugiyama review of covariate shift](http://www.ms.k.u-tokyo.ac.jp/sugi/2012/CSAreview.pdf); [Tibshirani likelihood-weighted conformal notes](https://www.stat.berkeley.edu/~ryantibs/statlearn-s24/lectures/conformal_ds.pdf)
- Fellegi–Sunter: for a comparison outcome, \(m = P(\text{outcome} \mid \text{match})\), \(u = P(\text{outcome} \mid \text{non-match})\), and posterior odds \(=\) prior odds \(\times (m/u)\). The log weight \(\log_2(m/u)\) is an additive form of the same Bayes factor. Linacre’s reading: \(m\) tracks data quality among true matches; \(u\) tracks accidental agreement. Disagreement weight in the usual field-agreement parameterization is \(\log((1-m)/(1-u))\). — [Linacre, 2023](https://www.robinlinacre.com/m_and_u_values/); [Brown et al., BMC Med Res Methodol 2017, Fellegi–Sunter weights](https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-017-0370-0)
- Winkler’s EM estimates those weights, and a match-prior term, from unlabeled agreement patterns. Conditional independence factors \(m(\tau)=\prod_i m_i(\tau_i)\) and \(u(\tau)=\prod_i u_i(\tau_i)\). He states that the optimal rule depends on the true likelihood ratio, that the independence assumption is what usually makes estimation possible, and that if the assumptions fail the resulting weights need not be optimal. The census note’s EM is for a broader family than strict independence. The same biomedical review reports that EM estimates of \(m\) sometimes beat estimates from labeled clerical review, and also that independence is often false, in which case the fitted mixture need not recover the true match and non-match classes. — [Winkler, Census RR2000/05](https://www.census.gov/content/dam/Census/library/working-papers/2000/adrm/rr2000-05.pdf); [Brown et al. 2017](https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-017-0370-0)
- Mapping the judge onto that language: a veto has \(m_{\text{veto}} = f\) and \(u_{\text{veto}} = t\), so its Bayes factor is \(K = f/t\), not a function of the decoy rate.

### Inferences

- **Constant-likelihood update (the “trust the veto more” regime).** Assume \(f_g(p) = f_s(p)\) and \(t_g(p) = t_s(p)\), with \(f_s, t_s\) estimated on labeled US/India inside the same bin, and \(t_s \neq f_s\). Then the unlabeled rate identifies the bin prior:
  \[
  \hat\pi_g(p) = \frac{t_s(p) - r_g(p)}{t_s(p) - f_s(p)},
  \]
  provided this lies in \([0,1]\), i.e. \(r_g(p)\) lies between \(f_s(p)\) and \(t_s(p)\). If \(t_s > f_s\) (the judge vetoes non-matches more often than matches), a higher \(r_g\) forces a lower \(\hat\pi_g\). Posterior after a veto:
  \[
  \hat P(Y{=}1 \mid J{=}0, p, g) = \frac{f_s(p)\,\hat\pi_g(p)}{r_g(p)},
  \qquad
  \frac{\hat P(Y{=}1 \mid J{=}0, p, g)}{1 - \hat P(Y{=}1 \mid J{=}0, p, g)}
  = \frac{\hat\pi_g(p)}{1-\hat\pi_g(p)} \cdot \frac{f_s(p)}{t_s(p)}.
  \]
  The weight that multiplies prior odds is \(K_s = f_s/t_s\). It does **not** grow with the veto rate. Decoy density changes only the prior odds \((1-\delta)/\delta\). Relative to the source bin,
  \[
  \frac{O_g}{O_s} = \frac{\delta_s}{\delta_g} \cdot \frac{1-\delta_g}{1-\delta_s}.
  \]
  That ratio is approximately \(\delta_s/\delta_g\) only when decoy density is small. It is not a license to multiply the stacker’s judge coefficient by the conflict-rate ratio. In a saturated logistic bin, \(\mathrm{logit}\,P(Y{=}1 \mid J) = \mathrm{logit}\,\pi + \log(m_J/u_J)\). A pure prior change moves the intercept. The veto contrast \(\log(f/t) - \log((1-f)/(1-t))\) stays put.
- **When “more conflicts” means the judge broke.** If \(f\) rises, \(K = f/t\) moves toward 1 or beyond, and the veto should move the probability **less**. The constant-likelihood inversion will happily invent a collapsed \(\pi\) and then treat the veto as highly informative. That is the wrong direction. Saerens EM and BBSE, fed the judge output or a stacker that still contains the judge, do exactly this: they are derived under \(p(x \mid y)\) (hence \(p(J \mid Y)\)) fixed, so every extra veto is read as a prior shift.
- **Which object each method is allowed to change.**
  - Known-prior / Saerens / Elkan-style reweight: changes \(q(y)\) only. Needs invariant class-conditionals. If the classifier is already calibrated to the source, the whole update is the displayed E-step. Identified from unlabeled \(x\) only under that invariance plus calibration (EM) or plus an invertible confusion matrix (BBSE). BBSE’s \(w\) does not need calibration; plugging \(w\) into the E-step formula does. Lipton’s weighted ERM needs source labels and does not single out one coefficient.
  - Platt \((A,B)\): \(A\) is a slope on the score (how informative the score is), \(B\) is an intercept. A pure prior shift, with the score’s class-conditional unchanged, belongs in the intercept once the score is otherwise calibrated; refitting \(A\) would fake a change in the likelihood ratio. Neither parameter is an MLE on unlabeled data, because the Bernoulli likelihood needs \(Y\). Temperature \(T\) only rescales logits. It is not a prior correction, and \(T\) is fit on labeled NLL. Using a larger \(T\) to “down-weight the judge” changes the slope, which is the unidentified reliability parameter, and still needs labels.
  - Covariate-shift importance weighting: assumes \(P(Y \mid X)\) is **the same** on the target, so the Bayes-optimal prediction given \((p, J, \text{other})\) does not change at all. For a correctly specified conditional model the population coefficients are invariant; the weights only change the finite-sample objective. There is no closed form that adjusts the judge coefficient alone. If the judge’s false-veto rate changed, \(P(Y \mid X)\) changed, the covariate-shift assumption is false, and reweighting does not identify the new coefficient.
- **Anchor the prior on a score that does not use the judge, then test the judge.** Run Saerens (if the no-judge score is calibrated on the source) or BBSE (if it is a hard classifier with invertible source confusion) using only \(p\) and the competition features. Call the result \(\pi_g^{\mathrm{anchor}}(p)\). Assumptions: \(P(\text{no-judge score} \mid Y)\) is invariant across groups, and for EM the score is calibrated to the source posterior. That anchor is **not identified** without labels; leave-one-country-out between US and India can falsify it where labels exist and cannot certify it for France. Given the anchor, constant judge likelihood predicts
  \[
  \hat r_g(p) = f_s(p)\,\pi_g^{\mathrm{anchor}}(p) + t_s(p)\,(1-\pi_g^{\mathrm{anchor}}(p)).
  \]
  Agreement with the observed unlabeled \(r_g(p)\) is the constant-likelihood regime: keep \(K_s = f_s/t_s\), use the new prior. A large excess of vetoes is the broken-judge regime.
- **What remains identified after the judge breaks.** Given an anchor \(\pi_g(p)\) and \(t_g \in [0,1]\),
  \[
  P(Y{=}1 \mid J{=}0, p, g) = 1 - \frac{t_g(p)\,(1-\pi_g(p))}{r_g(p)}
  \in \left[1 - \frac{1-\pi_g(p)}{r_g(p)},\; 1\right]
  \]
  whenever \(r_g(p) \le \pi_g(p)\) (so a zero true-veto rate is still feasible). The Bayes factor \(f_g/t_g\) is **not** point-identified: \(f_g\) is squeezed near \(r_g/\pi_g\) when \(1-\pi_g \ll r_g\), but \(t_g\) is barely constrained and \(K\) can still range widely. The posterior probability is what gets pinned, because almost all observed vetoes must be false vetoes. Point-identifying \(K\) by assuming \(t_g = t_s\) and setting \(\hat f_g = (r_g - t_s(1-\pi_g))/\pi_g\) adds an assumption that unlabeled data do not test.
- **Illustrative arithmetic on the task-brief rates (unmeasured, no \(n\)).** The brief states a test veto rate at \(p \ge 0.999\) of about \(0.08\%\) in US/India versus \(11.6\%\) in France, and a separate US-only replica rate of \(1.54\%\) on a France-like slice, with every replica veto a true match. Treating \(0.08\%\) as a stand-in for \(f_s\) only because \(\pi \approx 1\) on the source (itself an assumption): even a judge who vetoes every non-match (\(t = 1\)) and keeps \(f = 0.0008\) cannot produce \(r = 0.116\) unless the non-match mass in the bin is at least about \((0.116 - 0.0008)/(1 - 0.0008) \approx 0.115\). So either more than about one in nine of the France \(p \ge 0.999\) pairs are non-matches, or \(f\) rose, or both. A nominal score of \(0.999\) cannot absorb an \(11.5\%\) non-match rate without the score itself being wrong on France. By contrast, a \(10\)–\(20\times\) rise from \(0.08\%\) lands near \(0.8\%\)–\(1.6\%\). Under \(t \le 1\) and \(f = 0.0008\) that only requires on the order of \(1\%\) non-matches in the bin, which a prior-odds shift can produce without any change in \(f\). Constant likelihood is a possible reading of the US/India \(10\)–\(20\times\) figure and is not a possible reading of the France \(11.6\%\) figure unless \(\pi\) in that bin is far below \(0.999\). If one **assumes** \(\pi \ge 0.999\) anyway, the France posterior bound is \(P(Y{=}1 \mid \text{veto}) \ge 1 - 0.001/0.116 \approx 0.991\). The veto should barely move a \(0.999\) score. The replica’s “all vetoes were true matches” agrees in direction but was measured at \(1.54\%\), on a different system, with labels the test set does not have. Do not substitute \(11.6\%\) for a measured \(f\), and do not substitute \(1.54\%\) for the test France rate.
- Clipping every in-country veto is not this update. On the source, \(f_s\) near the observed \(0.06\%\)–\(0.08\%\) (task brief) makes \(K_s\) small if \(t_s\) is not also tiny, so those rare vetoes are strong. The brief states that clipping them costs accuracy; that cost is unmeasured in the papers reviewed, and it is what the constant-\(K\) formula predicts.

### Gaps

- The Saerens et al. 2002 PDF and the Esuli, Molinari, and Sebastiani TOIS 2020 reassessment of SLD did not load. The EM algebra above is taken from Alexandari et al.’s quotation of Saerens, not from the original PDF. Saerens’ abstract says they also give a test that the prior changed; the test statistic was not retrieved. Esuli et al.’s empirical criticisms were not retrieved.
- \(t_s(p)\) at \(p \ge 0.999\) is likely estimated from almost no labeled non-matches. The brief does not report that count. Then \(K_s = f_s/t_s\) has an unstable denominator even on US/India, while \(f_s \approx r_s\) is the better-identified source quantity. The procedure should propagate that uncertainty rather than treat \(t_s\) as known.
- No France \(n\) is given, so “\(11.6\%\) versus \(0.08\%\)” cannot be turned into a calibrated \(p\)-value here.
- Whether \(P(p \mid Y)\) is the same in France as in US/India is not testable without labels. The anchor prior inherits that untested assumption.

## How do you tell “prior moved” from “judge broke” with no target labels?

### Takeaway

Unlabeled data can show that \(P(J \mid p, g)\) changed, and can reject a constant \((f,t)\) if an external anchor says \(\pi\) cannot have fallen enough. They cannot, by themselves, split a moderate rate increase into \(\Delta f\) versus \(\Delta\pi\). Source calibration supplies \(f_s\) and \(t_s\); the label-free check is whether the unlabeled veto-versus-\(p\) curve matches \(f_s \pi + t_s(1-\pi)\) for a prior \(\pi\) taken from a judge-free score.

### Cited Findings

- Under label shift, \(q(\hat y) = \sum_y p(\hat y \mid y)\, q(y)\). BBSE inverts that with a source confusion matrix. The associated test is a test of \(w = 1\) **assuming** label shift, not a test that label shift is true. If \(p(\hat y \mid y)\) has changed, the inversion is the wrong model and can still return a weight vector. — [Lipton, Wang, Smola 2018](https://ar5iv.labs.arxiv.org/html/1802.03916)
- EM label-shift estimates are consistent for the prior only when the class-conditionals are unchanged and the probabilities used as \(\hat p(y \mid x)\) are calibrated. Alexandari et al. show biased calibration systematically moves the EM prior. A reliability check on the **source** is therefore part of the estimator, not an optional diagnostic. Expected calibration error on labeled bins is \(\sum_m (|B_m|/n)\,|\mathrm{acc}(B_m) - \mathrm{conf}(B_m)|\). — [Alexandari et al. 2020](https://ar5iv.labs.arxiv.org/html/1901.06852)
- Margin Density Drift Detection tracks the fraction of unlabeled points that fall inside a classifier’s uncertainty band. It is unsupervised and is designed to fire when drift will hurt accuracy, but the signal is a single margin occupancy, not a decomposition into prior shift versus a change in \(P(y \mid x)\). — [Sethi and Kantardzic, arXiv:1704.00023](https://arxiv.org/abs/1704.00023)
- Record-linkage EM will re-estimate \(m\), \(u\), and the match proportion together from unlabeled comparison patterns, but only inside a mixture model. Winkler: if the conditional-independence (or other working) assumption is wrong, the weights need not be optimal. A single binary field does not supply enough independent cells for that mixture. — [Winkler, Census RR2000/05](https://www.census.gov/content/dam/Census/library/working-papers/2000/adrm/rr2000-05.pdf)
- Two label-free shift detectors that do **not** resolve this problem: a domain classifier on covariates detects \(P(x)\) changes and is weak for a pure prior shift; a black-box shift detector on prediction margins detects prior shift and is weak for covariate noise that does not move predictions. Neither identifies whether a judge’s class-conditional error changed. — [Maggio, 2020, summarizing domain-classifier vs black-box shift detection](https://medium.com/data-from-the-trenches/towards-reliable-ml-ops-with-drift-detectors-5da1bdb29c63)

### Inferences

- **Diagnostic that uses source labels plus unlabeled target scores (the one that matches this stacker).**
  1. On labeled source, inside bins of the judge-free score, estimate \(f_s(p)\), \(t_s(p)\), and the source reliability curve of that score. Bins with no non-matches do not yield \(t_s\).
  2. Leave-one-country-out between US and India: freeze \((f,t)\) from one country, put the other country’s unlabeled \(r(p)\) through the constant-likelihood inversion, and compare \(\hat\pi\) to that country’s true \(P(Y{=}1 \mid p)\). This is the only empirical check that “more vetoes meant more decoys, not a worse judge” in the brief’s \(10\)–\(20\times\) setting. It uses labels. It says nothing about France by itself.
  3. On unlabeled group \(g\), compute \(r_g(p) = \widehat P(J{=}0 \mid p, g)\) and a judge-free anchor \(\pi_g^{\mathrm{anchor}}(p)\) by Saerens or BBSE.
  4. Reject constant likelihood if \(r_g(p)\) is outside \([\min(f_s,t_s), \max(f_s,t_s)]\), or if \(r_g(p) > f_s(p)\,\pi_g^{\mathrm{anchor}}(p) + 1 \cdot (1-\pi_g^{\mathrm{anchor}}(p))\). The second threshold does not need a usable \(t_s\). It is the right check at \(p \ge 0.999\), where labeled non-matches are scarce. If it fires, do not run EM on the judge.
  5. If \(r_g(p)\) matches \(f_s \pi^{\mathrm{anchor}} + t_s(1-\pi^{\mathrm{anchor}})\) within binomial error, keep \(K_s\) and update only the prior. “Trust the veto more” here means a lower posterior, not a larger coefficient.
- **Label-free signals that do not resolve the regime.**
  - A shift in the marginal of \(J\), or in \(J \mid p\), is visible with no labels (two-sample test on the veto indicator stratified by \(p\)). It is necessary but not sufficient: both a prior change and a broken judge move \(r_g(p)\).
  - Concentration of conflicts in one country is the same fact, sliced by \(g\). It localizes the shift. It does not say whether France’s \(11.6\%\) (task brief) is decoys or false vetoes.
  - BBSE weights that come out negative, or greater than any plausible prior ratio, are a red flag that the linear system is a bad description (sampling noise or a violated label-shift assumption). A tidy positive \(w\) is not evidence the assumption is true. Running BBSE on a score that includes the judge will report a prior shift in the broken-judge regime.
  - MD3 will often miss a judge that vetoes confidently: those rows need not sit in the margin. A rise in margin mass also does not say which feature caused it.
  - Source ECE of the judge estimates \(f_s\) and \(t_s\). It is not a target-domain calibration. Target ECE needs target labels.
  - Unsupervised Fellegi–Sunter EM on the full comparison vector can move \(m\) and \(u\) for every field, including a judge bit, but it confounds \(m_{\mathrm{judge}}\), \(u_{\mathrm{judge}}\), and the match prior. With one binary judge and no invariance assumption on \(p\), the confounding is exact: one equation, three unknowns. EM does not become identified just because the veto rate changed.
- **Reading the two task-brief regimes (inference, not a fit).** US/India at a \(10\)–\(20\times\) conflict increase can still sit inside \((f_s, t_s)\) if \(t_s\) is not small, so the constant-likelihood story is the one Bayes supports **if** LOCO says \(f\) and \(t\) actually traveled between those two countries. France at \(11.6\%\) versus \(0.08\%\) fails the \(t \le 1\) bound unless the anchor \(\pi\) is abandoned. The replica (labeled, \(1.54\%\), all vetoes true matches) is direct evidence that a US-trained judge’s \(f\), not just \(\pi\), moved on that France-like slice. It is not a measurement of test-France \(f\).

### Gaps

- No retrieved paper gives a labeled-free test that point-identifies \(f_g\) and \(\pi_g\) separately for one binary side feature.
- Rabanser et al., “Failing Loudly” (dataset-shift detection benchmarks) was not re-read; it is not cited above.
- Binomial tests need per-country, per-bin \(n\), which the brief does not give.
- The replica and the test judge are different numbers (\(1.54\%\) vs \(11.6\%\)). Nothing retrieved says they share an \(f\).

## Which published group-conditional trust methods fit on source labels plus an unlabeled target?

### Takeaway

Train-time dropout of the judge feature can be fit on source labels alone and is what makes a later hard mask well-defined; it does not learn a France-specific weight. Learning-to-defer and a supervised mixture-of-experts gate need labels for the expert’s correctness, including in any group whose reliability you want to estimate. Meta-distillation from a mixture of experts is the reviewed method that actually adapts to an unseen domain with unlabeled target data, but it adapts a whole predictor, assumes several labeled source domains, and does not identify a single likelihood ratio. Context-aware stacking gates found here are trained with labels.

### Cited Findings

- Learning to defer trains on tuples \((x_i, y_i, m_i)\): covariates, **target label**, and expert label. The Bayes rejector compares the classifier’s class probability to \(P(Y = M \mid X = x)\), the probability the expert is correct, which is learned from agreement with the target. With partial expert coverage the method still uses target labels on the rows that lack expert labels; it imputes expert error, it does not drop \(Y\). — [Mozannar and Sontag, ICML 2020](https://ar5iv.labs.arxiv.org/html/2006.01862)
- The same paper’s related-work pointer for “downstream expert / reject the expert” is Madras, Pitassi, and Zemel (the predict-responsibly deferral model) and selective classification, which sets a reject cost or a reject-rate constraint against labeled risk and does not use an external expert. — [Mozannar and Sontag 2020](https://ar5iv.labs.arxiv.org/html/2006.01862)
- Later deferral variants still use labels. Expert-agnostic L2D models expert competence from annotations, adds a Bayesian prior on that competence, and claims better generalization to experts whose behavior was not in training; it reduces annotation demand and does not claim a fully unlabeled target. A training-free conformal deferral method (author-reported) cuts expert labels by up to \(91.3\%\) on CIFAR-10H and HAM10000 and still needs labels to build conformal sets. Probabilistic L2D (ICLR 2025) uses EM for **missing expert annotations**, not for missing task labels. — [EA-L2D, arXiv:2502.10533](https://arxiv.org/html/2502.10533v2); [Bary, Macq, Petit, arXiv:2509.12573](https://arxiv.org/pdf/2509.12573); [Nguyen, Do, Carneiro, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/78df0f831fbe5854349dbdfccde7ee5d-Abstract-Conference.html)
- Meta-DMoE meta-trains a student to distill a mixture of domain-specific experts and, at test time, adapts that student on unlabeled target examples. Reported gains are versus other unlabeled-target adapters (ARM and variants) on iWildCam, Camelyon17, and FMoW. The meta-training domains are labeled. This is domain-level adaptation of a predictor, not an update of one feature’s \(m\) and \(u\). — [Zhong et al., arXiv:2210.03885](https://ar5iv.labs.arxiv.org/html/2210.03885)
- DART’s “dropout” drops previously added trees when the next tree is fit, to fight over-specialization of late trees. It is not input-feature dropout. Column subsampling per tree (`colsample_bytree` / `colsample_bylevel`) is the closer gradient-boosting analogue of input dropout, and it still uses the selected features on every row of that tree. — [Rashmi and Gilad-Bachrach, AISTATS 2015](https://arxiv.org/abs/1505.01866); [explanation distinguishing DART from column subsampling](https://stats.stackexchange.com/questions/200396/dropout-regularization-in-gbm)
- Neural dropout randomly zeros units or inputs **during training** to stop co-adaptation, and is not applied as a fresh mask at inference (weights are scaled instead). — [Dropout, Wikipedia, citing Srivastava et al.](https://en.wikipedia.org/wiki/Dropout_(neural_networks))
- CORE-STACK+ (preprint, submitted 2026-09-22) learns a small per-sample gate over ensemble aggregators, plus redundancy filtering and a ridge blender. The gate is meta-learned on labeled prediction pools. Author-reported benchmark gains are not independently replicated here. The paper does not claim an unlabeled-target, group-specific feature likelihood. — [CORE-STACK+, arXiv:2609.26905](https://arxiv.org/abs/2609.26905)

### Inferences

- **Can be fit from source labels, and then applied to an unlabeled group, without estimating a new reliability parameter:** train-time random missingness of the judge column (input dropout, not DART). The boosting path for “judge absent” is then on-support. A hard mask at inference is justified only to the extent it matches that training missingness. It implements “ignore the judge,” not “trust it 30% as much.” Shrinking the judge toward missing with a France-only weight is a different estimator and is not what dropout fits. The brief’s claim that a stacker trained with the judge sometimes dropped degrades less than inference-only masking is consistent with this train/test mismatch; the magnitude is unmeasured in the sources above.
- **Cannot be fit for France without France labels (or labels of an exchangeable group):** the deferral rejector \(r(x)\), because its target is whether the judge is right; a mixture-of-experts gate whose supervision is which expert matched \(Y\); a country-specific stacker coefficient on the judge; Platt \(A,B\) or temperature on France; any ECE on France. An unseen country id has no labeled rows, so a gate that takes country identity as an input has an unidentified France parameter. LOCO between US and India estimates transport between those two label regimes only.
- **Uses source labels plus unlabeled target, but does not answer the one-feature question:** Saerens EM and BBSE (prior only; judge likelihood held fixed). Meta-DMoE (whole-model adaptation; needs several labeled source domains at meta-train time; does not output \(f_g, t_g\)). Covariate-shift importance weighting (assumes the conditional label does not change). Fellegi–Sunter EM (can move \(m\) and \(u\) without labels only if the comparison vector identifies the mixture; a single judge bit does not).
- **Context-aware stacking,** as in the CORE-STACK+ gate, was not found in a form that is trained from unlabeled target groups. A per-row gate trained on source labels can depend on features that correlate with country, and might generalize, but that is ordinary supervised generalization, not an estimate of France’s veto likelihood.
- No reviewed method both (i) needs only source labels plus unlabeled target scores and (ii) point-identifies a new judge coefficient under simultaneous prior shift and a change in \(f\). The identified objects are the constant-likelihood posterior, or the posterior **interval** once an anchor \(\pi\) is assumed.

### Gaps

- Searches for “forget-me-not” and “feature ablation as regularization” did not return a GBDT stacking method. The hits were a VAE anti-collapse penalty, a 2026 local-overfitting distillation paper, and diffusion unlearning. Nothing retrieved quantifies train-time feature dropout versus inference-only masking for gradient boosting.
- Jacobs / Jordan hierarchical mixture-of-experts was not re-read. The gate conclusion above is from what Meta-DMoE and the deferral papers actually optimize, not from a fresh reading of the 1991–1994 MoE papers.
- No 2018–2026 paper was found that sets a group-specific trust weight for one side feature of a stacker from unlabeled target rows when the group prior and the feature’s error rate may both have moved.

## A concrete procedure: raise decisive vetoes with decoy density in US/India, and lower trust in France

### Takeaway

Do not rescale the judge coefficient by the conflict ratio. On a group that passes the constant-likelihood check, shift the match prior (Saerens or BBSE on a judge-free score) and multiply by the source Bayes factor \(f_s/t_s\). On a group that fails it, stop using \(f_s/t_s\); the label-free output is an interval for \(P(\text{match} \mid \text{veto})\) that, at \(p \ge 0.999\) and a \(11.6\%\) veto rate, stays near the anchor prior. Any single France coefficient adds an assumption the unlabeled slice does not identify.

### Cited Findings

- Posterior odds \(=\) prior odds \(\times m/u\), with \(m = P(\text{comparison} \mid \text{match})\) and \(u = P(\text{comparison} \mid \text{non-match})\). — [Linacre 2023](https://www.robinlinacre.com/m_and_u_values/)
- The prior-ratio update used in step 2 is the Saerens E-step, which Alexandari et al. show is maximum likelihood when \(p(x \mid y)\) is invariant and the score is calibrated; BBSE is the alternative that replaces the calibration requirement with an invertible source confusion matrix and then either reweights the posterior the same way or refits by weighted ERM. — [Alexandari et al. 2020](https://ar5iv.labs.arxiv.org/html/1901.06852); [Lipton et al. 2018](https://ar5iv.labs.arxiv.org/html/1802.03916)
- A hard mask is on-support only if training sometimes dropped that input. Inference-only dropout is not the dropout estimator. — [Dropout, Wikipedia](https://en.wikipedia.org/wiki/Dropout_(neural_networks)); [DART is tree dropout, not this mechanism](https://arxiv.org/abs/1505.01866)

### Inferences

**Quantities that are actually computable**

- From labeled US/India, optionally leave-one-country-out: inside bins of a judge-free score \(s_0\) (stage-2 \(p\) plus competition features, **not** the raw marginal of \(p\) if those features are used), \(\hat f_s(s_0)\), \(\hat t_s(s_0)\), source \(\hat\pi_s(s_0)\), and a calibration map of \(s_0\). Binning on \(s_0\) is what avoids assuming the judge is independent of the other features given \(Y\). That conditional independence is false if the judge reads the same fields the matcher reads; imposing it double-counts.
- From unlabeled test rows in group \(g\): \(\hat r_g(s_0)\) and the histogram of \(s_0\). Nothing else about \(Y\).

**Procedure**

1. **Train the stacker so “no judge” is a real path.** On source labels, fit the gradient-boosted stacker on \((s_0 \text{ features}, J)\) with \(J\) set to missing on a random fraction \(\alpha \in (0,1)\) of rows (missing, not the code for “no”). \(\alpha\) is a training hyperparameter, not a France trust weight. DART does not do this step.
2. **Anchor the group prior without the judge.** On unlabeled group \(g\), take the model’s judge-missing score, or a model fit without \(J\), and apply Saerens EM if that score is calibrated on the source, otherwise BBSE. Output \(\pi_g^{\mathrm{anchor}}(s_0)\) and the global \(q_g(Y{=}1)\). Assumption (untested on France): \(P(s_0 \mid Y)\) does not depend on \(g\). If this is false, France’s anchor prior is wrong and every later bound moves with it.
3. **Regime check, still without target labels.**
   \[
   r_g^{\max}(s_0) = \hat f_s(s_0)\,\pi_g^{\mathrm{anchor}}(s_0) + 1 \cdot (1-\pi_g^{\mathrm{anchor}}(s_0)).
   \]
   Also require \(\hat r_g(s_0)\) to lie between \(\hat f_s(s_0)\) and \(\hat t_s(s_0)\) when \(\hat t_s\) is estimated from enough labeled non-matches to be stable.
   - Pass (US/India, if LOCO agrees): constant likelihood stands. A \(10\)–\(20\times\) rise in \(r\) is read as a rise in decoy density.
   - Fail (France, if \(\hat r\) is \(11.6\%\) at \(s_0 \ge 0.999\) and the anchor \(\pi\) is anywhere near \(0.999\)): do not use step 4a.

4a. **Constant-likelihood posterior (the US/India update).** Do not multiply \(K\) by the decoy ratio. With \(K_s(s_0) = \hat f_s(s_0)/\hat t_s(s_0)\),
    \[
    \frac{P_g(Y{=}1 \mid J{=}0, s_0)}{P_g(Y{=}0 \mid J{=}0, s_0)}
    = \frac{\pi_g^{\mathrm{anchor}}(s_0)}{1-\pi_g^{\mathrm{anchor}}(s_0)} \cdot K_s(s_0).
    \]
    The symmetric factor when the judge says yes is \((1-\hat f_s)/(1-\hat t_s)\), under the same invariance. Equivalently, invert \(\pi\) from \(r_g\) itself,
    \[
    \hat\pi_g(s_0) = \frac{\hat t_s - \hat r_g}{\hat t_s - \hat f_s},
    \]
    and use \(\hat P = \hat f_s \hat\pi_g / \hat r_g\). If this \(\hat\pi_g\) and \(\pi_g^{\mathrm{anchor}}\) disagree beyond sampling error, the constant-likelihood assumption and the anchor assumption cannot both be true; do not average them silently.
    Relative to the source, the prior-odds factor is \((\delta_s/\delta_g)\cdot((1-\delta_g)/(1-\delta_s))\). For a logistic stacker whose veto contrast is \(\beta_J\), the term that moves is the intercept (the prior), not \(\beta_J\). Raising \(\beta_J\) “in proportion to decoy density” is a different estimator and overstates the veto exactly when the math says the likelihood ratio was stable.

4b. **Broken-judge posterior (the France update).** Do not plug in \(K_s\). Under only \(t_g \in [0,1]\) and the anchor,
    \[
    P_g(Y{=}1 \mid J{=}0, s_0)
    \;\in\;
    \left[
      1 - \frac{1-\pi_g^{\mathrm{anchor}}(s_0)}{\hat r_g(s_0)},
      \; 1
    \right]
    \]
    when \(\hat r_g \le \pi_g^{\mathrm{anchor}}\). The lower endpoint is the most a veto is allowed to move the probability. Operational point inside the interval, with no further unidentified choice: emit the judge-missing stacker probability from step 1, which ignores the veto. That is a France-specific reduction relative to applying \(K_s\), and it is computable from unlabeled scores once the regime check fails. It is **not** a calibrated France posterior; it is the training-supported action inside an interval.
    Illustrative only, using the brief’s \(11.6\%\) and an **assumed** \(\pi \ge 0.999\): the interval is about \([0.991, 1]\). Ignoring the veto and applying a strong source \(K_s\) are far apart; the bound says the strong \(K_s\) is illegal under that anchor. If the anchor \(\pi\) is itself only \(0.9\), the interval collapses downward and a veto can again be informative. The unlabeled rate does not choose between those anchors.

**Terms that are not identified without a label (or without an extra assumption)**

- \(f_g\) and \(t_g\) separately, and therefore \(K_g = f_g/t_g\), and any France-only \(\beta_J\). Setting \(t_g = t_s\) identifies them and is not tested by \(r_g\).
- \(\pi_g^{\mathrm{anchor}}\), unless \(P(s_0 \mid Y)\) is invariant. Unlabeled \(s_0\) identifies this prior only under that assumption (Saerens additionally needs calibration; BBSE needs an invertible confusion matrix and label shift).
- Platt slope \(A\), temperature \(T\), and a covariate-shift-adjusted judge coefficient. The first two need \(Y\) in \(g\). The third assumes the conditional \(P(Y \mid X)\) did not change, which is the hypothesis under test.
- Target-group ECE, precision of the veto, and the replica statement “every veto was a true match.” Those are labeled facts. The replica does not label test France. The test quantity \(11.6\%\) is a marginal rate, not a false-veto rate.
- \(\hat t_s\) in the extreme bin, if the labeled source has essentially no non-matches there. Then step 4a’s \(K_s\) is unidentified even on US/India; step 4b’s \(t \le 1\) bound does not use \(t_s\) and is the one that remains.

**What not to do**

- Do not run Saerens or BBSE on a score that still contains \(J\), then treat the resulting prior shift as the France correction. That forces the broken-judge regime into the trust-more regime.
- Do not clip US/India vetoes because France over-vetoes. Under a small \(f_s\) those vetoes carry a large \(|\log K_s|\).
- Do not hard-mask \(J\) only at inference in a model that saw \(J\) on every training row. The missing path was never fit. Step 1 is what the brief’s “dropped during training” result is pointing at; the size of the harm is a task-brief claim, not a number from the papers.

### Gaps

- The numerical value of the dropout rate \(\alpha\), the bin edges, and the LOCO tolerance are policy choices. No paper reviewed estimates them for this matcher.
- The posterior interval in step 4b is only as sharp as the anchor. There is no label-free way, in the sources reviewed, to confirm the anchor on an unseen country.
- A fully probabilistic middle ground (a prior on \(t_g\) yielding a posterior mean for \(K_g\)) was not found as a published estimator for this stacker. Adding one would still be an assumption, not a measurement.
