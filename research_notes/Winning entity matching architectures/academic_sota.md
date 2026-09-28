# Supervised entity matching on dirty textual records (academic SOTA, 2023–2026)

Scope note for the report writer: almost every neural “SOTA” F1 below is **pairwise F1 on a pre-blocked candidate set**, usually a few hundred to a few tens of thousands of pairs. That is not end-to-end F1 on millions of business records. Where a paper changes the candidate generator, the split, or the serialization, its F1 is **not comparable** to the Ditto / Magellan tables even when the dataset name is the same. Conflicts are stated in place. No blog-only numbers are used as F1.

“SOTA” is not one model. The published systems that actually run at enterprise scale are **pipelines** (blocking or indexing, pairwise classification, sometimes clustering). The benchmark leaderboards are mostly a **single pairwise classifier** scored after someone else has already built the candidate set.

## What is the current best published matcher on dirty name/address or product-title ER?

### Takeaway

There is no single 2024–2026 model that is SOTA on every standard ER set. On the canonical **fully supervised Magellan splits**, the reference numbers are still **Ditto** (fine-tuned BERT-class cross-encoder, 2020): dirty Walmart–Amazon F1 **85.69**, textual Abt–Buy **89.33**, textual Company **93.85**. Later fully supervised gains on those same splits are small (about 0–1 F1) or come from papers whose numbers others have failed to reproduce. On **hard product titles** (WDC 80% corner cases) and on **schema-heterogeneous** records, 2024–2026 work moves the lead to **prompted or fine-tuned generative LLMs**, not to HierGAT, Sudowoodo, Unicorn, CSGAT, CollaborEM, or MERAI. **AnyMatch** (GPT-2, 124M, 2024) is the best *small* zero-shot transfer matcher in its own nine-dataset table (mean F1 **81.96**), second to MatchGPT+GPT-4 (**86.36**), but it loses by 15–25 F1 on the hard product sets. **No 2023–2026 paper found here establishes a public SOTA on dirty business-name-plus-street-address**; the closest large name-like result is MERAI on voter files (a classical feature pipeline, not a transformer), and Ditto’s 96.5 F1 on two private company tables of 789k and 412k records (2020).

### Cited Findings

- Ditto (Li, Li, Suhara, Doan, Tan; arXiv:2004.00584, VLDB) casts EM as sequence-pair classification and fine-tunes BERT, DistilBERT, or RoBERTa, with optional domain-knowledge injection, TF-IDF summarization, and span/attribute augmentation. On ER-Magellan, full Ditto vs the previous best deep or classical matcher (DM+) and vs the Magellan classical matcher (numbers Magellan reports are those reproduced from Mudgal et al., the DeepMatcher paper):
  - **Structured:** Amazon–Google Ditto **75.58** (DM+ 70.7, Magellan 49.1, 11,460 pairs); Beer **94.37** (DM+/Magellan 78.8, 450 pairs); DBLP–ACM **98.99** (DM+ 98.45, Magellan 98.4, 12,363); DBLP–Google/Scholar **95.60** (DM+ 94.7, Magellan 92.3, 28,707); Fodors–Zagats **100** (Magellan 100, 946); iTunes–Amazon **97.06** (DM+ is Magellan at 91.2, 539); Walmart–Amazon **86.76** (DM+ 73.6, Magellan 71.9, 10,242).
  - **Dirty** (values randomly moved across attributes): DBLP–ACM **99.03** (DM+ 98.1, Magellan 91.9); DBLP–Google **95.75** (DM+ 93.8, Magellan 82.5); iTunes–Amazon **95.65** (DM+ 79.4, Magellan 46.8, 539 pairs); Walmart–Amazon **85.69** (DM+ 53.8, Magellan 37.4, 10,242). Ditto’s average degradation on the four dirty sets is **0.57** F1; DM+ degrades by **8.21**.
  - **Textual:** Abt–Buy **89.33** (DM+ 62.8, Magellan 43.6, 9,575); Company **93.85** (DM+ 92.7, Magellan 79.8, 112,632). The same paper’s baseline transformer **without** summarization scores **41.00** on Company, worse than Magellan.
  - **WDC products** (title/description/brand/spec; test 4,400 pairs, 27.27% positive): Ditto all-categories F1 **94.08** on the xLarge train set (214,736 pairs) vs DeepMatcher **90.16**. Category xLarge: Computers 95.45 vs 90.80, Cameras 93.78 vs 89.21, Watches 96.53 vs 93.45, Shoes 90.11 vs 92.61 (Ditto loses on Shoes).
  - Separate from the public Company benchmark: on two company tables of **789k and 412k records**, Ditto reports F1 **96.5%**. That figure is not a public Magellan split.
  — [Ditto PDF](https://arxiv.org/pdf/2004.00584)

- MatchGPT extended study (Peeters and Bizer; arXiv:2310.11244v4, 18 Oct 2024). Zero-shot **best prompt per model**, F1, on WDC Products (hard, 80% corner cases; test **259 positive / 989 negative**), Abt–Buy, Walmart–Amazon, Amazon–Google, DBLP–Scholar, DBLP–ACM:
  - GPT-4: **89.61, 95.78, 89.67, 76.38, 89.82, 98.41**
  - GPT-4o: **87.64, 93.95, 86.65, 73.56, 89.76, 97.06**
  - Llama-3.1 (70B-class chat, local): **83.67, 89.84, 84.85, 73.99, 86.32, 98.81**
  - GPT-4o-mini: **81.15, 91.93, 86.58, 72.18, 86.11, 97.60**
  - In-domain fine-tuned **RoBERTa**: **77.53, 91.21, 87.02, 79.27, 93.88, 99.14**
  - In-domain **Ditto** (RoBERTa-base inside Ditto): **84.90, 91.31, 86.39, 80.07, 94.31, 99.00**
  - Delta of best LLM minus best PLM: WDC **+4.71**, Abt–Buy **+4.47**, Walmart–Amazon **+2.65**, Amazon–Google **−3.69**, DBLP–Scholar **−4.49**, DBLP–ACM **−0.33**.
  - The same Ditto/RoBERTa models, trained on another dataset and tested on WDC (“unseen” products), collapse: Ditto unseen F1 **48.74 / 31.55 / 33.12 / 32.82 / 29.00** depending on the source set, i.e. drops of about **36 to 56** F1 versus in-domain Ditto at 84.90. RoBERTa unseen drops are about **22 to 61** F1.
  - Prompt choice moves GPT-4 on WDC from the low 80s to **89.61** (domain-complex-free). There is no single best prompt across model/dataset pairs.
  — [MatchGPT PDF](https://arxiv.org/pdf/2310.11244) ; earlier short version, zero-shot ChatGPT **82.35** F1 on a 50-product downsampled hard WDC set, vs RoBERTa fine-tuned on 2k pairs **82.72** and on the 20k set **89.32** — [Using ChatGPT for Entity Matching](https://arxiv.org/html/2305.03423)

- AnyMatch (Zhang, Groth, Calixto, Schelter; arXiv:2409.04073, Sep 2024; also AAAI workshop). Zero-shot **leave-one-dataset-out** fine-tune of **GPT-2 (124M)** with AutoML hard-positive filtering, attribute-level augmentation, and a 1:2 positive:negative cap. Test sets follow MatchGPT: if a test set exceeds 1,250 pairs it is cut to at most **250 positives and 1,000 negatives**. Mean F1 **81.96** vs MatchGPT+GPT-4 **86.36** (stated as within **4.4%** and **3,899×** cheaper per 1,000 tokens). Per-dataset F1, column order Abt–Buy, Amazon–Google, Beer, DBLP–ACM, DBLP–Google, Fodors–Zagats, iTunes–Amazon, Walmart–Amazon, WDC, mean:
  - AnyMatch: **86.05, 55.08, 96.55, 93.61, 90.59, 100.00, 90.91, 61.51, 63.31, 81.96**
  - MatchGPT GPT-4: **94.40, 74.91, 69.57, 95.60, 87.22, 97.67, 82.35, 89.67, 85.83, 86.36**
  - MatchGPT Mixtral-8x7B mean **64.26**; SOLAR mean **65.37**; Beluga2 mean **69.10**; GPT-3.5-Turbo-03 mean **73.12**; GPT-3.5-Turbo-06 mean **71.98**.
  - Their re-trained Ditto (BERT, 110M, **no** domain-knowledge tags, same leave-one-out protocol) mean F1 **66.05**, more than 15 points behind AnyMatch. Jellyfish-13B mean **77.83**, but it had seen six of the nine datasets in training, so those cells are not zero-shot.
  - AnyMatch is best on Beer, Fodors–Zagats, and iTunes–Amazon (all tiny) and second on Abt–Buy and DBLP–Google. It is far behind GPT-4 on Amazon–Google (**55.08 vs 74.91**), Walmart–Amazon (**61.51 vs 89.67**), and WDC (**63.31 vs 85.83**).
  — [AnyMatch HTML](https://arxiv.org/html/2409.04073v1) ; [AnyMatch abs](https://arxiv.org/abs/2409.04073)

- EDBT 2025 cross-dataset study (same AnyMatch authors; DOI 10.48786/edbt.2025.75) evaluates eight matchers on 11 datasets and reports that fine-tuned small models can match prompted large models, that data-centric tuning beats model-centric tuning, and that **AnyMatch[LLaMA3.2] is on par with trillion-parameter GPT-4** on their average. The page summary names Ditto, Unicorn, AnyMatch[GPT-2], AnyMatch[T5], AnyMatch[LLaMA3.2], and MatchGPT variants. **Per-dataset F1 from that table was not extracted** (PDF 404 at the open-proceedings URL tried). — [EDBT 2025 abstract page](http://dx.doi.org/10.48786/edbt.2025.75)

- Fine-tuning LLMs for EM (arXiv:2409.08185, 2024). On a **different** WDC construction (small test **500 pos / 4,000 neg**; product inputs often **title only**), standard fine-tuning helps smaller models and is mixed for larger ones. Readable within-paper WDC figures: gpt-4o-mini zero-shot about **81.61**, rising to **83.41** or **84.38** after fine-tuning depending on the explanation/selection setting; one comparison says a full GPT-4o fine-tune **drops** to **83.20** at much higher cost. A body sentence in the HTML version also states a GPT-4o fine-tune of **87.07** on WDC (+5.43). Those two GPT-4o figures were **not aligned to the same table row** here; do not average them. Llama-3.1-8B zero-shot on their WDC column is **53.36** and standard fine-tune on WDC reaches **69.19** — well below the in-domain Ditto number in MatchGPT, on a non-identical test. Fine-tuning helps in-domain transfer and **hurts cross-domain** (products ↔ scholar). — [Fine-tuning LLMs PDF](https://arxiv.org/pdf/2409.08185)

- Qwen3 factorial study (Zhang, Li, Calixto, Groth, Schelter; arXiv:2607.24688, 27 Jul 2026). 1,215 fine-tunes: three architectures (bi-encoder, cross-encoder, generative) × three Qwen3 variants × three sizes (0.6B, 4B, 8B) × nine datasets (Magellan, Machamp, WDC). Macro test F1, best variant per architecture:
  - Bi-encoder: 0.6B **72.5**, 4B **79.0**, 8B **81.5** (SBERT baseline **72.9**)
  - Cross-encoder: 0.6B **83.0**, 4B **84.1**, 8B **84.1** (their Ditto baseline macro **73.2**, high seed variance)
  - Generative: 0.6B **84.4**, 4B **86.8**, 8B **87.7** (zero-shot GPT baseline macro **82.0**)
  - The generative macro lead is **concentrated on two heterogeneous Machamp sets**. Excluding Semi-Heter and Semi-Rel, cross-encoder macros are **84.4** and generative **86.3 / 86.9** (paper text). On stable schemas the cross-encoder is essentially tied and much cheaper.
  - Per-dataset F1 (mean over five seeds), selected cells — Ditto vs cross-encoder 4B/8B vs generative 8B vs zero-shot GPT:
    - Abt–Buy: Ditto 84.7±3.5; cross 0.6/4/8B **93.0 / 94.0 / 94.9**; gen 8B **94.4**; GPT **87.2**
    - Amazon–Google: Ditto 72.2±1.2; cross 8B **78.6**; gen 8B **80.6**; GPT **64.6**
    - DBLP–Scholar: Ditto 94.9; cross 8B **95.2**; bi-encoder 8B **95.3**; gen 8B **95.0**; GPT **87.4** (saturated)
    - Walmart–Amazon: Ditto 85.5±1.6; cross 4B **91.6** / 8B **90.7**; gen 8B **90.6**; GPT **78.2**
    - WDC 20% corner cases: Ditto 76.4; cross 8B **88.7**; gen 8B **88.9**; GPT **87.3**
    - WDC 80% corner cases: Ditto **57.9±21.4** (unstable); cross 8B **83.2**; gen 8B **83.9**; GPT **82.2**
    - Semi-Heter (books, 12/12 attrs, heterogeneous, only 1,240/414/414 pairs, 38.2% positive): Ditto **56.3**; cross 8B **65.1±14.4**; gen 8B **91.5**; GPT **94.0**. Gap is mostly recall (cross-encoder recall stays ~43–51; generative 8B recall **84.8**). A rebalanced Semi-Heter makes cross-encoder recall ~96–97 and F1 comparable, so part of the gap is a class-conditional attribute shift, not “attention vs generation” in the abstract.
    - Semi-Rel (movies, 8 vs 14 attributes): cross 8B **85.0±11.4** (drops from 0.6B **97.6**); gen 8B **90.0**; Ditto **92.6±3.6**; GPT **87.2**
  - Throughput: 0.6B cross-encoder macro F1 **83.0** at about **473 pairs/s**; 8B bi-encoder reaches only **81.5** F1 and about **181 pairs/s**. Scaling the cross-encoder from 4B to 8B adds ~0 macro F1. 0.6B cross-encoder is already the practical default on the seven non-heterogeneous sets.
  - These Ditto numbers are **not** the 2020 Ditto table (Abt–Buy 89.33, Amazon–Google 75.58, Walmart–Amazon 86.76, WDC-all 94.08). Same names, different splits and a reimplementation. Use within-paper deltas only.
  — [Qwen3 EM factorial PDF](https://arxiv.org/pdf/2607.24688) ; Qwen3 weights are Apache-2.0 — [Qwen3 technical report](https://arxiv.org/abs/2505.09388)

- ComEM (Wang et al.; arXiv:2405.16884v3, 12 Dec 2024) is a **pipeline**, not a single matcher: Sparkly retrieves 10 candidates per query record, then a ranker (they use Flan-T5-XL) plus a selector LLM. Evaluation is **400 queries** (300 with a match) → **4,000 pairs**, not the official Magellan test files. Sparkly recall@10 on their eight clean-clean sets is **86.57% to 99.96%**. On that protocol, mean F1:
  - Supervised Ditto **80.69**, HierGAT **83.34** (Yao et al. 2022), Sudowoodo **70.32**
  - GPT-4o-mini matching **67.80**, 6-shot matching **77.94**, comparing **84.36**, selecting **82.26**, ComEM **86.42** (cost $0.09 in their units)
  - GPT-3.5 ComEM **85.61**
  - Walmart–Amazon cell that must not be compared to Ditto 2020: ComEM’s Ditto **57.75** vs HierGAT **78.55** vs GPT-4o-mini ComEM **88.56**. Original Ditto on the Magellan structured Walmart–Amazon file is **86.76** and on the dirty file **85.69**. The ComEM drop is a different candidate distribution (top-10 Sparkly neighbors), not a refutation of the 2020 table.
  — [ComEM HTML](https://arxiv.org/html/2405.16884v3)

- Unicorn (Tu et al., PACMMOD 2023, DOI 10.1145/3588938) is a multi-task encoder plus mixture-of-experts, not a dirty-data specialist. Reported F1 vs Ditto: Walmart–Amazon **86.93** vs 86.76; DBLP–Scholar **96.22** vs 95.6; Fodors–Zagats **97.67** vs 100; iTunes–Amazon **98.18** vs 97.06; Beer **87.5** vs 94.37. It wins some structured sets by under 1 F1 and loses Beer by ~7. — [Unicorn PDF](https://dl.acm.org/doi/pdf/10.1145/3588938)

- Sudowoodo (Wang et al., contrastive self-supervision; ICDE/Megagon, cited as 2023) is a **label-efficient** method. With 500 labels it reports up to **16%** F1 over Rotom and can beat Ditto **at the same small label budget**. It is not the fully supervised leader. An aggregator lists semi-supervised Abt–Buy around **81.1–81.7** F1 at budget 500, which is below full-data Ditto **89.33** and is a different training regime. — [Megagon summary](https://megagon.ai/sudowoodo-data-integration-apps/) ; [aggregator, not a primary table](https://www.wizwand.com/sota/entity-matching-on-ab-test)

- HierMatcher (Fu et al., IJCAI 2020), the hierarchical token/attribute/entity model that later HierGAT work builds on, dirty Walmart–Amazon F1 **68.5** vs DeepMatcher **53.8** vs Magellan **37.4**. That is well below Ditto’s later **85.69** on the same dirty file. Do not treat HierGAT/HierMatcher as the dirty-data SOTA after Ditto. — [HierMatcher PDF](https://www.ijcai.org/proceedings/2020/0507.pdf)

- CSGAT (Scientific Reports, 2025, DOI 10.1038/s41598-025-11932-9): Amazon–Google F1 **65.88±0.85** (Magellan 49.1 in their table) and Beer **82.73±1.22** (Magellan 78.8). Both are far below Ditto (75.58 and 94.37). Not SOTA. — [CSGAT PDF](https://www.nature.com/articles/s41598-025-11932-9.pdf)

- CollaborEM, in the unsupervised matching section of Zeakis et al. (VLDB Journal, published 4 Dec 2024): average recall **0.87**, average precision **0.26**, average F1 **0.35**, behind an unsupervised Sentence-BERT nearest-neighbor baseline around **0.57–0.58**. Not a supervised SOTA. — [Zeakis et al., VLDBJ](https://link.springer.com/article/10.1007/s00778-024-00879-4)

- MERAI (arXiv:2508.03767, 5 Aug 2025) is an **enterprise pipeline** (blocking/indexing, pairwise classical similarities such as Smith–Waterman and Monge–Elkan, clustering into disjoint cliques), compared with **Dedupe and Splink**, not with Ditto. On North Carolina voter dedup/linkage (name/address-like civil records, not Magellan):
  - Dedup F1: 2010 MERAI **97.4** (P 100 / R 95.0) vs Dedupe **85.7** vs Splink **82.1**; 2015 **94.1** vs 88.0 vs 70.6; 2020 **96.7** vs 94.6 vs 86.1.
  - Linkage F1: 2010–2015 **98.3**, 2010–2020 **97.3**, 2015–2020 **96.2**, all with precision **92.6–96.7** and recall **98.2–100**. Dedupe linkage precision is only **71.6–73.7** at recall ≥99.4. Splink linkage F1 **83.2–89.4**.
  - Scale: Dedupe failed past ~2–3 million records; MERAI ran experiments to **15.7 million** records and a bank deployment to **33 million**. Indexing emitted **82M–254M** pairs (e.g. 2020 dedup **208.7M** pairs → **4.4M** classified matches → **2.3M** cliques). This is the only 2025 system in this review that is actually about million-record name-like resolution, and its matcher is not a language model.
  — [MERAI HTML](https://arxiv.org/html/2508.03767v1)

- A 2026 Bamberg thesis compiles published BERT-era F1 and **warns that several of those cells have not been reproducible** (citing Low et al., 2024). Treat the following as secondary, not as re-measured SOTA: structured Amazon–Google SupCon 76.14 / AttendEM 77.67 / Ditto 75.58; textual Abt–Buy SupCon **94.29** vs Ditto 89.33; dirty Walmart–Amazon AttendEM **86.29** vs Ditto 85.69. JointBERT in that table is **below** Ditto (DBLP–Scholar structured 93.99, Abt–Buy 83.44, Company 91.40). The SupCon Abt–Buy jump of ~5 F1 over Ditto should not be quoted as established SOTA without the original table plus the failed-reproduction note. — [Bamberg thesis PDF](https://fis.uni-bamberg.de/bitstreams/841b3b7a-9870-4155-aaa8-5aa50a0ff1ea/download)

- EMTransformer: no 2023–2026 primary table under that name was found that beats Ditto on dirty or product benchmarks. The name is used loosely for the Ditto-style transformer matcher.

### Inferences

- If the question is “best **published fully supervised** matcher on the **official dirty Magellan candidate files**,” the honest answer is still **Ditto (2020)**, with only fractional later gains (Unicorn +0.2 on structured Walmart–Amazon, AttendEM +0.6 on dirty Walmart–Amazon in a secondary compilation). Those fractional gains are smaller than split and implementation noise.
- If the question is “best matcher on **hard product titles** under a zero-shot or low-label regime,” **GPT-4 prompted as in MatchGPT (2023–2024)** leads the small downsampled sets (WDC 89.61, Abt–Buy 95.78, Walmart–Amazon 89.67). Among open models that can be fine-tuned, the **July 2026 Qwen3 study** is the strongest head-to-head: an **8B generative** Qwen3 reaches WDC-80% F1 **83.9** and Walmart–Amazon **90.6**, and a **4B cross-encoder** is enough when the schema is stable (Walmart–Amazon **91.6**).
- If the question is “dirty **business name + address**,” the literature reviewed here does **not** have a transformer SOTA. Use MERAI-style precision/recall on voter-scale data only as a pipeline baseline against Splink/Dedupe, and Ditto’s private 96.5 only as an existence proof that a cross-encoder was run on ~1M company records in 2020.
- Benchmark SOTA and “wins on clean structured tables” diverge on purpose. DBLP–ACM/Scholar and Fodors–Zagats are saturated near 95–100 for both Magellan and transformers. Product titles, dirty attribute placement, and Machamp semi-structured books/movies are where architecture still matters.

### Gaps

- Original JointBERT, SupCon, AttendEM, KAER, and ROBEM papers were not re-opened; only the Bamberg compilation and the reproducibility warning.
- EDBT 2025 Table 3 (AnyMatch-LLaMA3.2 vs GPT-4 averages) was not extracted number-by-number.
- No public dirty **street-address** benchmark with a 2023–2026 transformer F1 was found. Company in Magellan is long company **text**, not an address field.
- ComEM, MatchGPT, AnyMatch, and the 2026 Qwen study use **different pair sets** that share dataset names. A merged “leaderboard” would be false.

## Which of those beat a strong GBDT on hand-built string features, and by how much, on dirty vs structured data?

### Takeaway

On the Magellan numbers that Ditto reprints from the DeepMatcher evaluation, classical Magellan (similarity features + a classical learner; **not** a named LightGBM/XGBoost rerun) is already near-optimal on **clean structured citation and restaurant** data, and transformers add roughly **0–6 F1**. On **dirty** and **textual product** data the same learner collapses, and Ditto’s gain is **+14 to +49 F1** versus Magellan and **+13 to +32 F1** versus the best pre-transformer neural matcher. Anything that matches or beats Ditto on those dirty files also beats that GBDT-style baseline by about the same margin. A 2024 embedding study finds Magellan’s average F1 only **0.60** across its five sets, **27.5%** relative below deep models, with the gap largest on the least linearly separable set. No 2024–2026 paper in this review re-benchmarks LightGBM or XGBoost head-to-head with Qwen3 or MatchGPT on the dirty splits.

### Cited Findings

- Magellan F1 as reported in Ditto’s appendix (source column: “Magellan (reported in [34])”, i.e. Mudgal et al. 2018), side by side with full Ditto. Positive rate of the candidate sets is about **9.4%** (Walmart–Amazon) to **25%** (Company), so these are already blocked pairs, not all-pairs.
  - **Structured, small gap or tie:** Fodors–Zagats 100 vs 100; DBLP–ACM 98.4 vs 98.99 (**+0.6**); DBLP–Scholar 92.3 vs 95.6 (**+3.3**); iTunes–Amazon 91.2 vs 97.06 (**+5.9**, 539 pairs); Beer 78.8 vs 94.37 (**+15.6**, 450 pairs — Magellan was the previous best, ahead of DeepMatcher).
  - **Structured, large gap:** Walmart–Amazon 71.9 vs 86.76 (**+14.9**); Amazon–Google 49.1 vs 75.58 (**+26.5**). Amazon–Google is the structured exception: string-feature linearity fails even without injected dirt.
  - **Dirty, Magellan falls over and Ditto does not:** DBLP–ACM 91.9 vs 99.03 (**+7.1**; Magellan itself dropped 6.5 from the clean version); DBLP–Scholar 82.5 vs 95.75 (**+13.3**); iTunes–Amazon 46.8 vs 95.65 (**+48.9**, and Magellan dropped **44.4** from its clean 91.2); Walmart–Amazon 37.4 vs 85.69 (**+48.3**, Magellan dropped **34.5** from 71.9). DeepMatcher+ on dirty Walmart–Amazon is only **53.8**, so the transformer, not “deep learning” in general, supplies most of the dirty-data gain (**+31.9** vs DM+).
  - **Textual:** Abt–Buy 43.6 vs 89.33 (**+45.7**); Company 79.8 vs 93.85 (**+14.1**). Without summarization the transformer baseline is **41.00** on Company, **below** Magellan — hand-built similarities can beat a naive long-sequence BERT.
  — [Ditto PDF](https://arxiv.org/pdf/2004.00584)

- Zeakis et al., VLDB Journal (4 Dec 2024), on their embedding benchmark (not a one-for-one reprint of the 13 Magellan files): Magellan average F1 **0.60**, range **0.37** (DSM5, low linearity) to **0.91** (DSM3, high linearity), **27.5%** below deep learning on average, but **+55%** versus GloVe on DSM1 and sometimes far ahead of frozen embeddings. RoBERTa, once fine-tuned, is the best dynamic encoder, within **0.5%** F1 of the per-dataset best on average. Frozen BERT is poor. — [Zeakis et al.](https://link.springer.com/article/10.1007/s00778-024-00879-4)

- HierMatcher (2020) already showed the dirty-data pattern before Ditto, with smaller absolute numbers: dirty Walmart–Amazon Magellan **37.4**, DeepMatcher **53.8**, HierMatcher **68.5**. Structured Walmart–Amazon Magellan **71.9** vs HierMatcher **81.6**. — [HierMatcher PDF](https://www.ijcai.org/proceedings/2020/0507.pdf)

- CSGAT (2025) beats the Magellan numbers it cites on Amazon–Google (**65.88** vs 49.1) and Beer (**82.73** vs 78.8) but loses to Ditto on both, so it does not change the “who beats GBDT” conclusion. — [CSGAT PDF](https://www.nature.com/articles/s41598-025-11932-9.pdf)

- The 2026 Qwen3 study does **not** include a GBDT arm. Its Ditto reimplementation is the classical-neural reference, and even that reimplementation is weak on WDC-80% (**57.9±21.4**) relative to a 4B–8B Qwen3 cross-encoder (**81.7–83.2**). That is evidence about transformer families, not about LightGBM. — [arXiv:2607.24688](https://arxiv.org/pdf/2607.24688)

### Inferences

- **Dirty vs structured is the whole story.** A strong similarity-feature learner is a legitimate production baseline on clean, aligned, high-overlap fields (citations, restaurants). It is not competitive once values move across columns or the match lives in long product text. The largest published gaps versus Magellan are on dirty iTunes–Amazon and dirty Walmart–Amazon, and on textual Abt–Buy.
- Beer, Fodors–Zagats, and iTunes–Amazon are **under 1,000 pairs**. A +15 F1 on Beer should not be used to justify an architecture for millions of addresses.
- Later LLM papers usually **assume** the GBDT has already lost, and compare to Ditto or RoBERTa. Where they still lose to in-domain Ditto (MatchGPT on Amazon–Google and DBLP–Scholar), they would not be expected to overturn the structured-data conclusion that classical features remain close.
- Do not describe the Magellan column as “LightGBM” or “XGBoost” unless a specific codebase says so. In these papers it is “Magellan’s classical ML matcher on similarity features.”

### Gaps

- No primary 2023–2026 table was found that reports LightGBM or XGBoost F1 on dirty Magellan next to Ditto, AnyMatch, or Qwen3.
- Zeakis et al. use DSM\* dataset codes; the 0.60 average was not mapped back to Abt–Buy / Walmart–Amazon / Company in the text extracted here.
- Whether a tuned GBDT on character n-gram and address-component features would close more of the dirty **address** gap than it does on dirty **product titles** is not answered by these benchmarks.

## What is the winning recipe for one pairwise model of at most 8B parameters under an Apache or MIT license?

### Takeaway

When labeled match/non-match pairs exist for the **target** schema, the winning published recipe inside the license cap is a **fine-tuned cross-encoder**, not a prompted 70B model and not a bi-encoder. A **RoBERTa or BERT cross-encoder in the Ditto setup** (MIT / Apache-2.0, well under 1B) is still the right default on stable Magellan-style files. If a larger open model is allowed, **Qwen3-4B as a cross-encoder (Apache-2.0)** matches Qwen3-8B on macro F1 (**84.1**) and beats that paper’s Ditto reimplementation by about **11** macro F1, with the cleanest gains on Abt–Buy, Walmart–Amazon, and hard WDC. Switch the **same Qwen3 4B–8B checkpoint to a generative yes/no head** only when attributes are heterogeneous or shifted (Semi-Heter: generative 8B **91.5** vs cross-encoder 8B **65.1**). **AnyMatch’s GPT-2 (MIT, 124M)** is the right *zero-shot transfer* recipe when there are **no** target labels, but it is not the winner once target labels exist, and it fails the hard product sets. **Llama-3.1/3.2 at 8B is out of scope** for this constraint (Llama community license, not Apache/MIT), even though EDBT 2025 and the fine-tuning paper show it can be strong. Do not deploy Mixtral, Flan-T5-XXL, or GPT-4o-mini under a “one model ≤ 8B, Apache/MIT” rule.

### Cited Findings

- License facts used for the cut: Qwen3 open weights, including the 0.6B–8B dense models and the embedding models, are **Apache-2.0**. — [Qwen3 technical report](https://arxiv.org/abs/2505.09388) ; [Qwen3 README](https://github.com/QwenLM/Qwen3/blob/bac19d60/README.md?plain=1) ; [Qwen3-Embedding](https://cdn.jsdelivr.net/gh/yanfeng98/paper-is-all-you-need/papers/00073-Qwen3-Embedding.pdf). BERT and DistilBERT are Apache-2.0; RoBERTa and DeBERTa and GPT-2 are MIT (standard Hugging Face licenses; Ditto and AnyMatch explicitly fine-tune BERT and GPT-2). Llama-3.1-8B and Llama-3.2, used by the fine-tuning paper and by AnyMatch’s EDBT extension, are **not** Apache/MIT. GPT-4, GPT-4o, and GPT-4o-mini are proprietary APIs, not weight-licensed models under 8B.

- Cross-encoder vs bi-encoder vs generative, same Qwen3 family, supervised pairwise labels (arXiv:2607.24688):
  - Joint pair encoding beats independent encoding at every size. An **8B bi-encoder (81.5)** still loses to a **0.6B cross-encoder (83.0)**.
  - For cross-encoders, **4B equals 8B** at macro F1 **84.1**. Extra size mostly helps the bi-encoder (+6.5 from 0.6B to 4B, +2.5 from 4B to 8B) and the generative model (+2.4 then +0.9).
  - Generative and cross-encoder are close on in-domain structured sets (Walmart–Amazon cross 4B **91.6** vs gen 8B **90.6**; Abt–Buy cross 8B **94.9** vs gen 8B **94.4**; DBLP–Scholar all ~**95**). Generative wins under attribute-combination shift (Semi-Heter **91.5** vs **65.1**).
  - Zero-shot GPT macro **82.0** beats their Ditto **73.2** but loses to a fine-tuned 0.6B generative Qwen3 **84.4** and to a 0.6B cross-encoder **83.0**. Target-task fine-tuning of a small Apache model beats a much larger prompted proprietary model **on average in this study**, with the proprietary model still ahead on Semi-Heter (**94.0** vs generative 8B **91.5**).
  - Practical operating point they state: on the seven datasets excluding Semi-Heter and Semi-Rel, the **4B cross-encoder** is the favorable accuracy/throughput point; 8B adds little accuracy. The 0.6B cross-encoder already does **83.0** F1 at ~**473 pairs/s**.
  — [arXiv:2607.24688](https://arxiv.org/pdf/2607.24688)

- In-domain small cross-encoders vs large prompted models, MatchGPT Table 4 (not all of these models satisfy the license cap; the comparison is about the recipe): in-domain Ditto **beats GPT-4** on Amazon–Google (**80.07 vs 76.38**), DBLP–Scholar (**94.31 vs 89.82**), and DBLP–ACM (**99.00 vs 98.41**), and **loses** on WDC (**84.90 vs 89.61**), Abt–Buy (**91.31 vs 95.78**), and Walmart–Amazon (**86.39 vs 89.67**). The same Ditto, without target labels, falls to the **30s–40s** F1 on WDC. So the recipe is “fine-tune on target pairs,” not “use the biggest model.” — [MatchGPT PDF](https://arxiv.org/pdf/2310.11244)

- AnyMatch shows the complementary case: **no target labels**, one MIT model at 124M, transfer data from the other public sets. Mean F1 **81.96** vs a similarly trained Ditto **66.05** vs GPT-4 **86.36**. The cost claim (3,899× cheaper than GPT-4 per 1,000 tokens, four orders of magnitude fewer parameters) is the deployment argument. The quality claim fails on exactly the dirty product-title sets a business matcher cares about (Amazon–Google **55.08**, Walmart–Amazon **61.51**, WDC **63.31**). — [AnyMatch](https://arxiv.org/html/2409.04073v1)

- LoRA / PEFT vs full fine-tune: the 2026 Qwen paper uses the PEFT library’s default recommendation for some runs and also **fully fine-tunes** the 0.6B models. A clean LoRA-only vs full-fine-tune F1 table for 4B–8B was **not** extracted. The fine-tuning-LLM paper’s Llama-8B results are full-model or provider-default fine-tunes and are **license-ineligible**. Structured explanations added about **+5** F1 for Llama-8B on their WDC column (69.19 → 74.13 in one table) and under **+1** for gpt-4o-mini (83.41 → 84.38). That is a data recipe, not a reason to pick an 8B generative model over a 4B cross-encoder when labels match the test schema. — [arXiv:2409.08185](https://arxiv.org/pdf/2409.08185) ; [arXiv:2607.24688](https://arxiv.org/pdf/2607.24688)

- ComEM’s best open-source **prompted** numbers (not fine-tuned, and several models exceed 8B): Mistral-7B selecting-strategy mean F1 **73.80** vs matching **45.91**; Flan-T5-XXL (11B) selecting **77.72**. Both are below in-domain Ditto’s historical product F1 and below Qwen3-0.6B cross-encoder macro **83.0**. Prompt pipelines do not replace a fine-tuned cross-encoder when labels exist, and ComEM itself uses **two** models. — [ComEM HTML](https://arxiv.org/html/2405.16884v3)

- Ditto’s own serialization lesson still binds small encoders: on the long Company file, turning off summarization drops F1 from **93.85** to **41.00**, below Magellan **79.8**. A ≤8B cross-encoder still needs truncation, TF-IDF or field selection, and explicit column markers. Domain-knowledge span highlighting is **not** available in the zero-shot setting AnyMatch enforces. — [Ditto PDF](https://arxiv.org/pdf/2004.00584)

### Inferences

- Recommended single model under the constraint, in order:
  1. **Target labels, stable schema (aligned name/title/address fields):** fine-tune one cross-encoder. Start with **RoBERTa-base or DistilBERT (Ditto recipe, <1B, MIT/Apache)**. Move to **Qwen3-4B cross-encoder (Apache-2.0)** if the 2026 within-study gains on product titles replicate on your split. Do not spend the step from 4B to 8B on a cross-encoder; the paper’s macro gain is ~0.
  2. **Target labels, dirty or shifted fields** (values in the wrong column, semi-structured vs structured, the Machamp pattern): fine-tune **Qwen3-4B or 8B as a generative matcher** (yes/no token), same Apache license. The 8B generative model is the one place size still pays (macro 86.8 → 87.7, and Semi-Heter 88.8 → 91.5).
  3. **No target labels:** **AnyMatch-style GPT-2 (MIT, 124M)** trained on other labeled EM corpora, and expect to lose to a proprietary LLM on hard product titles. A Qwen3-0.6B generative model **fine-tuned on other EM sets** is the natural 2026 upgrade of that idea, but the EDBT “LLaMA3.2 matches GPT-4” result uses a **non-Apache** base, so it is not the licensed recipe. This review did not find an Apache 4B–8B leave-one-out table that reproduces the 81.96 mean.
  4. **Bi-encoder:** use it for blocking, not as the final judge. Even at 8B it trails a 0.6B cross-encoder.
- Threshold the cross-encoder score on a validation set if precision matters. The 2026 paper already selects τ on validation; that is the precision knob, not a different architecture.
- None of these F1 numbers were measured on millions of pairs. A 4B cross-encoder at a few hundred pairs per second is an inference-budget choice, not evidence that pairwise F1 survives a 10^12 all-pairs space. The matcher recipe assumes a blocker in front (next section).

### Gaps

- No head-to-head of **LoRA-only Qwen3-4B/8B** vs a full fine-tune vs Ditto on the **official** dirty Magellan files.
- DeBERTa is an eligible MIT cross-encoder and is a common EM backbone, but no 2023–2026 table in this review isolates DeBERTa-v3 against RoBERTa on dirty Walmart–Amazon.
- Apache-licensed generative models other than Qwen3 (for example Mistral-7B **fine-tuned**, as opposed to prompted) were not found with a full Magellan F1 table.
- Whether Qwen3-4B cross-encoder F1 on Walmart–Amazon (**91.6** in the 2026 study) exceeds original Ditto (**86.76**) **on the official file** is unproven; the splits differ.

## Blocking and retrieval that feed the matcher: what recall and reduction are reported at million-record scale?

### Takeaway

At million-record scale the best-documented result is still **syntactic blocking and sparse nearest-neighbor search, tuned to a recall floor**, not a dense bi-encoder with HNSW/ScaNN. In Papadakis et al.’s dirty synthetic scale test (**10k to 2 million** entities), Standard Blocking keeps the best recall/precision balance (pair completeness **0.81** and precision about **0.55–0.60** at 2M). Dense **FAISS and ScaNN get faster but lose recall** (about **0.67** and **0.70** at 2M, precision about **0.02**). DeepBlocker does not get past **100k** entities in that study. Sorted Neighborhood was tried and **dropped because it consistently lost**. Sparkly’s recall@10 of **86.57–99.96%** is on small clean-clean benchmarks, not on millions of records. MERAI shows that a hand-built index on voter data can cut a multi-million-record job down to **~10^8** candidate pairs and then to **~10^6** matches, but it does not publish a pair-completeness/reduction-ratio pair in the Papadakis sense. Splink is a matcher/linker more than a retrieval SOTA; on MERAI’s data its precision is the weak point.

### Cited Findings

- Papadakis, Fisichella, Schoger, Mandilaras, Augsten, Nejdl, “How to reduce the search space of Entity Resolution: with Blocking or Nearest Neighbor search?” (arXiv:2202.12521, journal version VLDB Journal 2024, “Open benchmark for filtering techniques”). Methods: blocking workflows (standard blocking, q-grams, extended q-grams, suffix arrays, and variants), string similarity joins, sparse kNN joins, MinHash / cross-polytope / hyperplane LSH, **FAISS**, **ScaNN**, and **DeepBlocker**. They optimize each method for a **pair-completeness (recall) target, typically PC ≥ 0.9**, then compare precision (pairs quality) and runtime. Schema-agnostic settings are more robust to missing and misplaced values than schema-based ones. Similarity thresholds produce worse candidate sets than **cardinality** thresholds. Syntactic token representations beat semantic embeddings for **filtering**, because ER vocabularies are full of out-of-vocabulary and domain tokens; semantic embeddings help the **matcher**, not the blocker, except on two of their real sets.
  - Real clean-clean sets (not millions of records): after tuning, FAISS and ScaNN usually sit just above the 0.9 recall target, but precision is often poor (examples from their Table VIII: FAISS/ScaNN PC **0.900–0.971**, while PQ falls to **0.001**, **1.5×10^−4**, or **0.004** on some sets, and is high only on easy ones such as **0.93**).
  - LSH candidate reduction versus brute force, average over the sets in their Table VI: MinHash LSH **48%**, cross-polytope LSH **91%**, hyperplane LSH **89%**. They explicitly say this is still worse than blocking and sparse NN, because a 48–91% cut of an all-pairs space can remain a huge set.
  - **Sorted Neighborhood was implemented and then omitted from the tables because it consistently underperformed** the blocking workflows they do report.
  - Dirty synthetic scale (40% duplicates, 10k → 2M entities):
    - Standard Blocking: PC **0.91** at 10k down to **0.81** at 2M, precision held around **0.55–0.60**. Best balance in the blocking family.
    - Q-gram blocking: PC stays **≥ 0.9**, but precision falls from **0.44** at 10k to **~0.001** at 2M.
    - Extended q-gram blocking at 2M: PC **0.86**, PQ **0.03**.
    - Suffix-array style workflows: PC falls from **0.91** at 10k to **~0.55** at 2M.
    - Sparse kNN-join: PC down to **0.81** at 2M; the paper says ε-join and kNN-join keep **perfect precision** on these synthetic sets (strong claim; it is about their synthetic dirty generator, not about business addresses).
    - **ScaNN at 2M: recall 0.70, precision 0.02. FAISS at 2M: recall 0.67, precision 0.018.** FAISS is the fastest method that finishes 2M (**1.3 hours**); ScaNN takes **3.2 hours**. Runtime growth from 10k to 2M is ~**700×** for FAISS and ~**1,600×** for ScaNN, versus ~**8,000–20,000×** for blocking workflows, but blocking does not give up as much recall.
    - DeepBlocker and its DDB variant stop at **D100k** (quadratic memory; they cite ~300 GB for a 200k×200k float block). LSH variants hit the 0.9 recall target only by emitting too many candidates to reach the largest sets.
  - Conclusion they draw for large inputs: only blocking workflows, FAISS, and ScaNN finish all dirty synthetic sizes in under about 4 hours, and **FAISS wins speed while losing the recall/precision tradeoff**. The method they single out as combining a cardinality cutoff with a syntactic representation is **kNN-join**.
  — [arXiv:2202.12521](https://arxiv.org/abs/2202.12521) ; [VLDBJ 2024 abstract](https://link.springer.com/article/10.1007/s00778-024-00868-7) ; dataset sizes D1M = 1,000,000 entities and D2M = 2,000,000 are listed in the benchmark repo — [ContinuousFilteringBenchmark](https://github.com/gpapadis/ContinuousFilteringBenchmark)

- Sparkly (Paulsen et al. 2023), as used by ComEM: for each record in the smaller table, 10 nearest candidates from the larger table, **recall@10 between 86.57% and 99.96%** on eight clean-clean ER sets whose larger side is on the order of DBLP–Scholar / Walmart–Amazon (tens of thousands of records, not millions). That means a matcher evaluated only on those 10 neighbors is already blind to up to **13.4%** of true matches on the worst set. — [ComEM HTML](https://arxiv.org/html/2405.16884v3)

- Zeakis et al. (VLDBJ, 4 Dec 2024): among sentence-embedding blockers, **S-GTR-T5** is best on 7 of 10 datasets. They also run blocking-then-unsupervised-matching with k=10 and report end-to-end unsupervised F1 around **0.58**, not pair completeness at million scale. ZeroER did not finish within 6 hours on half their sets; the embedding blocker finished in under a minute. — [Zeakis et al.](https://link.springer.com/article/10.1007/s00778-024-00879-4)

- The 2026 Qwen3 matcher paper treats bi-encoders as the blocking-capable architecture (cached vectors, ANN) and cross-encoders as the pairwise judge. It does **not** report pair completeness or reduction ratio at a million records. — [arXiv:2607.24688](https://arxiv.org/pdf/2607.24688)

- MERAI indexing on multi-million voter files (absolute counts, not PC/RR): 2010 dedup **118.4M** indexed pairs → **1.5M** classified matches → **1M** cliques; 2020 dedup **208.7M** → **4.4M** → **2.3M**; linkage jobs **149M–254M** indexed pairs and **7.6M–14.1M** classified matches. Wall time on their hardware for the full MERAI pipeline at 2M records is about **1:06**, versus Dedupe **6:10**, with Dedupe’s clustering term dominating past 1M. They state Dedupe does not run past ~2–3M records, while MERAI was applied to **15.7M** experimental records and **33M** bank records. — [MERAI](https://arxiv.org/html/2508.03767v1)

- Splink (Linacre, Lindsay, Manassis, et al.) is the probabilistic Fellegi–Sunter linker MERAI uses as a baseline. On those voter tasks its dedup precision is **60–76.5%** at recall **85.7–98.4%**, and linkage precision **88.6–94.1%** at recall **78.3–85.2%**. MERAI’s authors call that unsuitable when false links are expensive. Splink is not a neural blocker and is not reported here with a PC/RR figure at 10^6 records. — [MERAI](https://arxiv.org/html/2508.03767v1)

- A Dec 2024 survey of blocking (Gurevich) taxonomizes logic, q-gram, Bloom-filter, LSH, and neural blockers (DeepER, DeepBlocker, transformer embeddings, neural LSH) but the extract used here is methodological, not a new million-scale PC table. — [Gurevich survey PDF](https://raw.githubusercontent.com/imvladikon/imvladikon/master/articles/entity_resolution_blocking_survey.pdf)

### Inferences

- For a million-record **name/address** job, the evidence-backed blocker is **schema-agnostic standard blocking or a syntactic cardinality kNN / q-gram join, tuned so pair completeness stays near 0.9**, accepting that precision of the candidate set may be terrible and must be repaired by the pairwise model. Dense ScaNN/FAISS are the latency play when a drop from ~0.9 to ~0.7 recall is acceptable, or when they are used only as a first stage with a high k and a second syntactic pass. A learned bi-encoder (DeepBlocker, Qwen3 bi-encoder) is not what the million-scale experiments crown.
- Do not quote Sparkly recall@10 or Magellan candidate-set F1 as if they were reduction ratios on 10^6×10^6.
- MERAI is the existence proof that **10^8 candidates from 10^7 records** is an operable reduction, but without a published pair-completeness number one cannot say what fraction of true address duplicates that index dropped.
- The pairwise SOTA models in the first section never see this candidate distribution. Dirty Magellan “candidate sets” are a few thousand labeled pairs with a 10–25% positive rate. A blocker at 2M entities with PQ ≈ 0.02 produces a radically harder negative distribution than those test files.

### Gaps

- No extracted PC and RR for **business name + street address** at 10^6 or 10^7 records.
- HNSW is not broken out separately from FAISS’s indexes; the paper’s fastest FAISS configuration discussed in the tuning advice is a **flat** index, while the 2M timing uses an approximate IVF-style index. Exact HNSW recall at 2M was not found.
- Sparkly has no number in this review at million-record scale.
- Papadakis et al. is 2022/2024 and uses fastText-style vectors for the dense methods, not 2024–2026 Qwen3 embeddings. A modern bi-encoder might move the FAISS/ScaNN recall, and this review did not find that experiment at 2M dirty records.
- “Perfect precision” for kNN-join on the synthetic sets should not be copied onto real addresses without the paper’s generator assumptions.

## Which papers optimize F-beta with beta < 1, or otherwise run precision-heavy ER?

### Takeaway

**No 2023–2026 entity-matching paper found in this review trains or selects models by F-beta with beta < 1.** The field reports pairwise F1. Precision-heavy behavior shows up as an **operating point**, not as a metric: MERAI’s voter dedup is run at **100% precision** and **92.5–95% recall** (F1 still the headline), and the 2026 Qwen3 cross-encoders hold high precision while recall is what fails under attribute shift. A 2015 practitioner’s note defines pairwise F-beta and says beta is “usually 1.” A 2025 vendor blog discusses F0.5 for identity resolution and is not a measured result.

### Cited Findings

- Ditto, MatchGPT, AnyMatch, ComEM, Unicorn, the Qwen3 factorial study, and Zeakis et al. all state that the reported metric is **F1**. MatchGPT says full precision and recall live in the project repository rather than in the paper tables. The Qwen3 study picks the decision threshold τ on validation data to maximize the usual F1-style cutoff and, in the Semi-Heter analysis, breaks errors into precision vs recall: the cross-encoder’s precision stays high while recall stays near **42.6 / 48.9 / 51.1** across 0.6B/4B/8B. That is evidence the model is already the more precision-heavy of the two heads, but they do not score F0.5. — [Ditto](https://arxiv.org/pdf/2004.00584) ; [MatchGPT](https://arxiv.org/pdf/2310.11244) ; [AnyMatch](https://arxiv.org/html/2409.04073v1) ; [Qwen3 EM study](https://arxiv.org/pdf/2607.24688)

- MERAI dedup is the clearest precision-heavy **system** result: precision **100%** on 2010 and 2015 dedup and **100%** on 2020, recall **95.0 / 92.5 / 93.7**, F1 **97.4 / 94.1 / 96.7**. Linkage is slightly less precision-heavy (precision **92.6–96.7**, recall **98.2–100**). They motivate this by “high-stakes” false links, then still publish F1. They do not mention F-beta. — [MERAI](https://arxiv.org/html/2508.03767v1)

- Papadakis et al. tune blockers to a **recall floor (PC ≥ 0.9)** and then rank by precision of the candidate set. That is recall-heavy blocking on purpose, the opposite of an F0.5 matcher, because misses in blocking can never be recovered. Standard Blocking is their precision-leaning blocker (PQ **0.55–0.60** at 2M with PC **0.81**). — [arXiv:2202.12521](https://arxiv.org/abs/2202.12521)

- Hand and others, “A Practitioner’s Guide to Evaluating Entity Resolution Results” (arXiv:1509.04238), define pairwise precision, recall, and F1, and mention a beta-weighted form in which beta < 1 emphasizes homogeneity over completeness. They say beta is usually set to 1. This is a metrics note, not a matcher that optimizes F0.5, and it is outside 2023–2026. — [ar5iv:1509.04238](https://ar5iv.labs.arxiv.org/html/1509.04238)

- Tilores’ 2025 blog explains F0.5 as the precision-weighted F-score for identity resolution. It is not a paper and contains no benchmark F0.5. Not used as a result. — [Tilores blog](https://tilores.io/content/precision-and-recall-in-entity-identity-resolution/)

- Fast beta linkage (Kundinger, Reiter, Steorts, Bayesian Analysis 2025, DOI 10.1214/24-BA1427) is a **Bayesian Fellegi–Sunter** speedup (“beta” is the beta prior, not F-beta). It does not optimize F0.5. — [Project Euclid](https://doi.org/10.1214/24-ba1427)

### Inferences

- A precision-heavy deployment on top of this literature is a **threshold change** on a cross-encoder or a **precision constraint inside blocking-plus-clustering** (MERAI), not a different network. The Qwen3 Semi-Heter breakdown is the warning: if you only watch precision, a cross-encoder can look safe while missing half of the true matches once fields move.
- Because candidate sets in Magellan are already enriched (roughly 10–25% positives), F1 on those files **under-penalizes** a matcher that would flood a real blocker output where positives are ≪ 1%. MERAI’s 100% precision at the cost of 5–7.5% missed duplicates is a more honest precision-heavy target for name/address dedup than any Magellan F1.
- If a report needs an F0.5 number, it will have to be **computed from precision and recall**, and those are often not in the papers (MatchGPT points at a GitHub repo; Ditto prints F1 only).

### Gaps

- No primary paper in 2023–2026 was found whose leaderboard, loss, or model selection uses F0.5 or F-beta with beta < 1.
- MatchGPT’s per-dataset precision and recall were not copied out of the GitHub repo; only F1 from the paper is cited here.
- MERAI does not say how the 100% precision operating point was chosen (validation sweep vs business rule), so it cannot be copied as an algorithm.
