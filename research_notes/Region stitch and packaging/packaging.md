# Packaging a stitched entity-resolution file under an open-set country rule

Research date: 2026-09-26. Scope is public reproducibility guidance (Kaggle winner-documentation guidelines, DrivenData prize-winner template, NeurIPS / Pineau and Papers with Code code checklists) applied to the binding contest rules in the research brief. No public page for this specific entity-resolution contest was found. Where a general contest norm conflicts with the open-set country rule or the ban on external lookups, the brief wins.

Binding rules used below (research brief; no public URL found):

- Treat country as an open set of string labels: do not hard-code, filter, or one-hot the pipeline to only {US, India}. Every test entity, including any entity whose country string never appears in training, must appear in the submission.
- The legal formulation is "countries that do not appear in the training labels", not a held-out country name. Train labels are on US and India; the test file adds one more country. That name must not be a branch condition.
- No external databases, geocoders, or lookups. Models must be Apache or MIT and at most 8B parameters.
- The final zip must contain `output/matching_results.tsv` (the scored file), `output/candidate_pairs.tsv` (the candidate set the model actually scored), code that regenerates both from the provided data, a README with the exact run, pinned requirements, and a methodology document with sections methodology, blocking, model and features, and other relevant information. Top teams' packages are reviewed before final rankings.
- Public and private leaderboards are two splits of one submitted file.
- The shippable file is a stitch: rows from one run where the country appears in the training labels; rows from another run where it does not. The second run is a baseline plus pairs added back where a judge veto is undone at a high model score, with each non-reference record assigned to at most one reference.
- Baseline packaging must reproduce an exact md5 from the stacker saved at upload time and from the judge-score files that existed when that file was built.

The official TSV column names were not in any source opened here. The specification uses roles (submission key, reference id, non-reference id, country string), not invented header names.

## What do organisers accept as reproducible for a stitched submission: one command that rebuilds the stitch from pinned artifacts, checked by md5, versus a notebook that cannot be re-run?

### Takeaway

A notebook that cannot be re-run is not an acceptable reproducibility artifact. The bar that Kaggle and DrivenData actually publish is a non-interactive entry point that regenerates the scored file from raw competition data plus pinned model weights, with exact dependency pins and the command written in the README. Neither guide requires an md5 of the output file. For this contest, exact md5 of both TSVs is still the right acceptance check, because the ranking is two splits of one file and the baseline must match the upload-time stacker byte for byte. The country set itself must be recomputed from the training file inside that command; it must not be frozen inside the pinned scores.

### Cited Findings

- Kaggle's winning-model guidelines say the core requirement is that the archive "detail all the pieces needed by the host to reproduce your solution with the score your team achieved on the leaderboard within a reasonable margin." The same page says models go in a single zip archive. — [Kaggle Winning Model Documentation Guidelines](https://www.kaggle.com/WinningModelDocumentationGuidelines)
- That page's README requirements that were visible in the page index include hardware (CPU specs, core count, memory, GPU specs and count), "important side effects" of the code (example given: preprocessing that overwrites the original data), and "key assumptions" (example given: the outputs folder must be empty before a training run). Numbered README items 2 through 5 were not in the retrieved index text. — [Kaggle Winning Model Documentation Guidelines](https://www.kaggle.com/WinningModelDocumentationGuidelines)
- The same guidelines require `requirements.txt` at the archive root with exact versions, illustrated as `pandas==0.23.0`, generatable with `pip freeze` in Python or `devtools::session_info()` in R. A Dockerfile may replace that file only if installs use exact version numbers. They also require `SETTINGS.json` as the only place that sets train, test, model, and output paths, and they require every I/O path to come from that file. They require a serialized trained model so prediction does not retrain, and `entry_points.md` listing commands, with training separated from prediction. The three example entry points are `python prepare_data.py` (read `RAW_DATA_DIR`, write `CLEAN_DATA_DIR`), `python train.py` (read cleaned train data, write `MODEL_DIR`, optional `CHECKPOINT_DIR`), and `python predict.py` (read cleaned test data, load `MODEL_DIR`, write `SUBMISSION_DIR`). They also require `directory_structure.txt` produced by `find . -type d` from the project root, and a subfolder of config files whose install locations are described in the README. — [Kaggle Winning Model Documentation Guidelines](https://www.kaggle.com/WinningModelDocumentationGuidelines)
- Standard Kaggle winner language, as used on the ICR competition rules, requires delivery of "the final model's software code as used to generate the winning Submission" and says that code "must be capable of generating the winning Submission" and must describe resources required to build and run it. The same rules say submissions "may not use or incorporate information from hand labeling or human prediction of the validation dataset or test data records." The NFL Big Data Bowl 2026 analytics rules use the same "capable of generating the winning Submission" sentence and point at the same documentation guidelines. — [ICR competition rules](https://www.kaggle.com/competitions/icr-identify-age-related-conditions/rules); [NFL Big Data Bowl 2026 analytics rules](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-analytics/rules)
- Kaggle's separate solution-writeup rubric (a public essay, not the winner zip) scores "Completeness" and "Strong Supporting Materials": enough detail that another data scientist can "understand exactly what was attempted and conceivably reproduce it." Required essay sections named there are context, overview of the approach (models, preprocessing, features, validation), details of the submission, and sources. That rubric is for a 2023 swag award based on write-up votes, not for prize-code review. — [Kaggle Solution Write-Up Documentation](https://www.kaggle.com/solution-write-up-documentation)
- DrivenData's prize-winner template says the goal is to "set up your solution as if it were a finished open-source project" with "an obvious point of entry" and that "the inference step [should] be fully reproducible so that it can be run with any new set of data, whether or not it is included in the test set." The README must explain how to produce the submission "starting from a fresh system with no dependencies installed." Trained weights must be included or linked so predictions can be made "without needing to retrain your model from scratch." Raw competition data need not be in the archive. — [drivendataorg/prize-winner-template](https://github.com/drivendataorg/prize-winner-template)
- DrivenData's README template says training instructions should use "a main point of entry such as an executable script that runs all steps of the pipeline in a deterministic fashion e.g., a `.py` file, `.sh` file, or Jupyter notebook." The inference section is stricter: "We should be able to rerun inference programmatically using only the raw competition data and your model weights. There should not be any retraining or manual input required." It also says "We only rerun the inference step, so we should be able to download any model files needed without rerunning the full training process," and asks whether training needs network access. The template asks for Python version, a `requirements.txt` or `environment.yml` with versions, expected directory layout before the run, hardware, and rough train and inference times. — [README_template.md](https://raw.githubusercontent.com/drivendataorg/prize-winner-template/main/README_template.md)
- The template's example submission is a command-line inference script with explicit paths (`python src/run_inference.py`), pinned via `requirements.txt`, not an interactive notebook. Its example README records hardware, training time, and inference time. — [example_submission/README.md](https://github.com/drivendataorg/prize-winner-template/blob/main/example_submission/README.md)
- A DrivenData code-competition runtime (DaT Parkinson's) builds the submission zip with `uvx rpzip` "to create deterministic archives" and validates them with `just check-submission`. That is a deterministic zip of `main.py` plus assets, not an md5 of a prediction TSV. Execution has no general internet access, so dependencies must already be in the runtime lockfile. — [competition-sfmn-parkinsons-runtime README](https://github.com/drivendataorg/competition-sfmn-parkinsons-runtime/blob/main/README.md)
- The NeurIPS code-submission guidelines say a code submission should include training code, evaluation code, and a specification of dependencies, and point to the Papers with Code guide. Code "ideally should be self-contained and executable"; if it is not, the authors must explain why. — [NeurIPS Code and Data Submission Guidelines](https://nips.cc/public/guides/CodeSubmissionPolicy)
- The current NeurIPS paper checklist asks whether code, data, and instructions needed to reproduce the main results are included, and says "The instructions should contain the exact command and environment needed to run to reproduce the results." It also asks whether training details were specified, including "data splits, hyperparameters, how they were chosen." Releasing code is not mandatory if another reasonable reproducibility route is documented; "no because the code is proprietary" is listed as an acceptable answer on that checklist. — [NeurIPS Paper Checklist](https://neurips.cc/public/guides/PaperChecklist)
- Pineau's Machine Learning Reproducibility Checklist v2.0 (7 Apr 2020) asks shared code to include a dependency specification, training code, evaluation code, pre-trained models, and a "README file [that] includes table of results accompanied by precise command to run to produce those results." For results it asks for the hyper-parameter range considered, "method to select the best hyper-parameter configuration," every hyper-parameter used, the exact number of training and evaluation runs, the measure reported, central tendency and variation, average runtime or energy, and the computing infrastructure. It also asks for train/validation/test split details and an explanation of any excluded data and preprocessing. — [ReproducibilityChecklist.pdf](https://www.cs.mcgill.ca/~jpineau/ReproducibilityChecklist.pdf)
- Papers with Code's ML Code Completeness Checklist, which NeurIPS 2021 recommended, is the same five items: dependencies (`requirements.txt`, `environment.yml`, or `setup.py`; Docker optional), training code, evaluation code, pre-trained models, and a README with a results table plus the precise command. They report that NeurIPS 2019 repos with all five items had the highest GitHub stars (median 196, mean 2,664). Hosting options they list for weights include Zenodo (DOI, versioning), GitHub Releases, and Hugging Face Hub. — [paperswithcode/releasing-research-code](https://github.com/paperswithcode/releasing-research-code/blob/master/README.md)

### Inferences

- "Reproducible" in these guides means a reviewer can execute a documented command and obtain the same submission (DrivenData: from raw data plus weights, no manual steps; Kaggle: code capable of generating the winning submission, score within a reasonable margin). It does not mean a narrative notebook, a screenshot of a leaderboard, or a pre-stitched TSV with no generator.
- A Jupyter notebook is explicitly allowed by DrivenData only as a deterministic, non-interactive entry point. A notebook that cannot be re-run fails DrivenData's "no manual input" sentence and fails Kaggle's "capable of generating the winning Submission" sentence. For this zip, ship one non-interactive command. A notebook may sit beside it as an explanation; it must not be the only way to rebuild the files.
- Kaggle's published success criterion is the leaderboard score "within a reasonable margin," not a byte-identical file. That is weaker than this contest needs. Public and private scores are two splits of one file, and the baseline must match the upload-time stacker md5. A retrain that lands "close" is a different submission. The stitch command should check md5 (or sha256) of `output/matching_results.tsv` and `output/candidate_pairs.tsv` against checksums recorded at upload time, and exit non-zero on mismatch. No organiser guide retrieved here states that md5 requirement; it is required by this contest's one-file scoring plus the upload-time stacker constraint, not by Kaggle's margin language.
- Pinned artifacts are what these guides expect for the model, not a substitute for the decision rule. Kaggle B7 and DrivenData both say to ship weights so inference does not retrain, and DrivenData says reviewers often rerun inference only. This contest also requires code that regenerates both TSVs from the provided data. The command that satisfies both is: read the provided training-label file and recompute the country set; read the pinned stacker and judge-score files (path plus content hash in the README); apply the two decision functions; write both TSVs; check hashes. A second documented command should show how those score files were produced from the provided data and the pinned model. Baking the stitch into a TSV and documenting it in prose does not regenerate anything.
- Do not describe the stitch as something that was clicked together in a notebook session. `entry_points.md` or the README should contain the exact command, the paths from a single settings file, and the expected checksums.

### Gaps

- The live Kaggle guidelines URL returned only a site shell on 2026-09-26. Quotations are from the search index of that URL. README items 2–5 were not in the indexed excerpts, so they are not stated here.
- No Kaggle, DrivenData, or NeurIPS page found in this pass requires an md5 of the prediction file. DrivenData's deterministic-zip note (`rpzip`) is about the archive bytes, not the TSV.
- DrivenData's `example_documentation_guide.pdf` is referenced by the prize-winner template and was not text-extracted.
- No public writeup found that documents a country-conditional stitch of two entity-resolution runs, so there is no organiser precedent specific to this stitch shape.

## How should the country rule be written so a reviewer sees an open set computed from the training file, not a hardcoded country name?

### Takeaway

Write the rule as a pure function of the training-label file: the set of country strings that occur on a row that has a label. The complement is every other country string, computed by membership, with no literal country name in the branch. A reviewer should be able to recompute the set, name the two decision functions, name the merge key, and check that every required id is present once.

### Cited Findings

- The binding rule is to treat country as an open set of string labels and not to hard-code, filter, or one-hot the pipeline to only {US, India}, and to keep every test entity in the submission. The formulation the team is allowed is countries that do not appear in the training labels. The held-out country name is not the rule. — Research brief (no public URL)
- DrivenData asks for inference that runs "with any new set of data, whether or not it is included in the test set," with no manual input. A country branch that only works for the countries present in this test file fails that generalisation even before this contest's open-set sentence is applied. — [README_template.md](https://raw.githubusercontent.com/drivendataorg/prize-winner-template/main/README_template.md)
- Pineau v2.0 asks for a clear explanation of assumptions, for train/validation/test split details, and for an explanation of any data that were excluded and of all preprocessing. A silent drop of rows whose country string is outside a fixed list is an undocumented exclusion under that checklist. — [ReproducibilityChecklist.pdf](https://www.cs.mcgill.ca/~jpineau/ReproducibilityChecklist.pdf)
- Kaggle's winner-doc guidelines tell authors to state key assumptions the code makes. An assumption that "country is one of a fixed pair of strings" would be an assumption a reviewer is told to look for, and it is forbidden by this contest. — [Kaggle Winning Model Documentation Guidelines](https://www.kaggle.com/WinningModelDocumentationGuidelines)

### Inferences

Reviewer-checkable specification. Column names below are roles. Map them to the official headers in the README; do not invent a second schema.

**Inputs**

- `train_labels`: the provided training file. A row counts as labeled only if it carries a supervision label (match / non-match, or a linked reference). Rows with a missing label do not contribute a country.
- `entities`: the provided entity table used at scoring time. Every entity has an id and a country string. This table is not a source of supervision.
- `required_ids`: the id list the scorer expects in `matching_results.tsv`. If that list is not shipped separately, it is the set of ids in the official submission skeleton provided with the data.
- `run_labeled`: the stacker decisions saved at upload time, for use only on countries inside the computed set. Identify the file by relative path and content hash.
- `run_unlabeled_scores`: the judge-score file (or files) that existed when the baseline was built: pair id, model score, and the judge veto bit. Identify each file by relative path and content hash.
- No test-label file, leaderboard file, gazetteer, or geocoder is an input.

**Set definition**

```text
C = unique country strings on train_labels rows whose supervision label is present
```

`C` is computed at runtime from `train_labels` only. It is not the literal `{US, India}`, not a one-hot vocabulary fitted with `handle_unknown="error"`, and not the list of countries observed in the test file. The log may print the computed value so a reviewer can see it. The branch may not compare a country string to a constant. The held-out country is handled only as `country ∉ C`. Any future string absent from `C` takes the same branch.

**Two decision functions**

- `D_labeled(id)`, used iff `country(id) ∈ C`: the decision row from `run_labeled` for that id. Do not refit this run inside the stitch step. The stitch must reproduce that run's bytes for these ids (the upload-time stacker md5, restricted to this subset, or the full-file md5 after the merge below).
- `D_unlabeled(id)`, used iff `country(id) ∉ C`: start from the baseline decision; undo a judge veto only for a pair that is already in the judge-score file and whose model score is at least `τ`; then assign each non-reference record to at most one reference. Tie-break must be deterministic and written down (for example highest score, then lexicographic pair id). Pairs that were never scored cannot be added back. `τ` is a scalar constant, not a function of the country string.

**Merge key**

- Join key: the submission id (the id the scorer joins on), not the country string and not a row number.
- Country for the branch is looked up from `entities` on that id. Do not trust a country column copied into an old score file if it can disagree with `entities`.
- Output row for id `i` is `D_labeled(i)` if `country(i) ∈ C`, else `D_unlabeled(i)`.
- Ids are not dropped, rewritten, or filtered because their country is outside `{US, India}`.

**Invariants a reviewer can check by script**

- Coverage and uniqueness: every id in `required_ids` appears in `matching_results.tsv` exactly once. If the submission key is the reference id, that is the "every reference id appears once" check. It is a key constraint, not a claim that one reference matches only one non-reference.
- Assignment: each non-reference id is assigned to at most one reference. A reference id may appear on more than one row when the format is one row per non-reference or per pair.
- Open set: the set of country strings on submission ids equals the set of country strings on `required_ids`. In particular it is not a subset of `C`.
- No literal branch: the stitch source, settings file, and methodology do not contain a condition on a specific country name. A comment that prints the observed value of `C` is not a branch. A test fixture may use synthetic country strings that are not the real held-out name.
- Superset: every pair that `matching_results.tsv` treats as a match is a row of `candidate_pairs.tsv`. Stated again in the checklist section.

**Dry-run that fails closed if the training country set changes**

- Recompute `C` from `train_labels`. If a recorded checksum of `C` is stored, it is an observation, not the definition. If the recomputed set differs from the set that was logged when the upload-time file was built, exit non-zero and do not emit the old TSV under the old md5.
- Fixture A: append a labeled training row with a new synthetic country string `S`. Ids whose country is `S` must switch from `D_unlabeled` to `D_labeled`. If they stay on the add-back policy, the test fails.
- Fixture B: remove all labels for one country that had been in `C`. Those ids must switch to `D_unlabeled`.
- Fixture C: an entity whose country string is neither in the original `C` nor equal to any string the authors have typed in the repo must still appear exactly once and must take `D_unlabeled`.
- The dry-run must not contain the real held-out country name. Use synthetic strings so the test does not teach the pipeline that name.

### Gaps

- The official column names and whether the scored file is one row per reference, one row per test entity, or one row per pair were not in the brief or in the public pages opened. The uniqueness check must be pointed at whatever column the scorer uses as the key. If "every reference id appears once" and "many non-references can share a reference" are both true, the file cannot be one row per pair keyed only by reference id; the README has to say which id is the unique key.
- No public source confirms the label column's null convention (which rows "have a label"). The definition above follows the brief's phrase "country strings that occur with a label."

## What goes in the methodology so an add-back is a property of "no training labels", and which sentences look like a rule violation?

### Takeaway

Describe the add-back as the policy for every country string that has no training label. Do not name the held-out country, do not cite a France-only resource, and do not hide how `τ` was chosen. A hand-built country gazetteer, a geocoder, or a threshold chosen on the public leaderboard should be written as such if that is what happened; those sentences read as violations of this contest even when the code is later cleaned. Renaming them to "no training labels" does not make them compliant.

### Cited Findings

- This contest forbids external databases, geocoders, and lookups, and forbids hard-coding country to the training pair. The methodology sections it requires are methodology, blocking, model and features, and other relevant information. Packages of top teams are reviewed before final rankings. — Research brief (no public URL)
- Pineau v2.0 requires the method used to select the hyper-parameter configuration, not only the final value, plus any assumption and any excluded data. — [ReproducibilityChecklist.pdf](https://www.cs.mcgill.ca/~jpineau/ReproducibilityChecklist.pdf)
- The NeurIPS paper checklist asks for hyperparameters and "how they were chosen," in the paper or the supplement. — [NeurIPS Paper Checklist](https://neurips.cc/public/guides/PaperChecklist)
- Kaggle rules on ICR forbid hand labeling or human prediction of validation or test records. The NFL Big Data Bowl 2026 analytics rules allow external data only when it is public and equally available to every participant, or meets that competition's reasonableness clause. That external-data permission conflicts with this contest's ban on external databases, geocoders, and lookups. This contest's ban wins. The same NFL rules also repeat the hand-labeling ban. — [ICR competition rules](https://www.kaggle.com/competitions/icr-identify-age-related-conditions/rules); [NFL Big Data Bowl 2026 analytics rules](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-analytics/rules)
- A secondary account of Kaggle's Don't Overfit II competition describes a winner who submitted each variable as its own submission to read off leaderboard feedback and then set coefficients. The chapter cites the winner's discussion post. That is evidence that public-leaderboard probing can extract test-set information. It is not a statement that every contest bans threshold search on the public score. — [The Kaggle Book, chapter excerpt (Packt)](https://www.packtpub.com/en-in/product/the-kaggle-book-9781801817479/chapter/introducing-kaggle-and-other-data-science-competitions-2/section/introducing-kaggle-ch02lvl1sec04); discussion URL cited there: https://www.kaggle.com/c/dont-overfit-ii/discussion/91766 (not opened in this pass)

### Inferences

Put the following in the four required sections. Use the set `C` from the specification above. Do not write the held-out country name anywhere in the methodology, README, comments, or fixtures.

**methodology**

- State the task as entity resolution over an open set of country strings.
- Define `C` in one sentence: the set of country strings that occur on a training row that has a label, computed from the training file, not a fixed list.
- State the stitch: if `country ∈ C`, use the upload-time labeled-country decisions; if `country ∉ C`, use the no-label policy. Say that this second policy is used because that country string has no training labels, so the labeled-country judge is not treated as calibrated there. Do not say it is used because of properties of a particular country (language, postal format, administrative geography).
- Define the no-label policy: baseline decisions; a judge veto is undone only when the model score on an already scored pair is at least `τ`; then each non-reference record is assigned to at most one reference, with the tie-break named.
- State the selection method for `τ` in the same place as the value. Acceptable provenance is a threshold fixed from training-country validation, or a predeclared constant chosen before the stitch, with the validation slice described. The method must not depend on a country name.
- State the invariants: every required id once, unseen country strings retained, each non-reference assigned at most once, one submitted file for both leaderboard splits.
- Say explicitly what was not used: no external database, no geocoder, no hand-built gazetteer, no country-name branch, no second file for the private split.

**blocking**

- Describe how candidate pairs were proposed from the provided fields only.
- Say whether country is used as a block key. If it is, it must be "same country string" or another rule that does not list allowed strings. An allow-list equal to the training countries is a filter the rules forbid.
- State that `candidate_pairs.tsv` is the set the model actually scored, and that every match in the stitched file, including a pair whose veto was undone, is in that set. A pair inserted only at stitch time, with no score in the judge-score file, is not a candidate the model scored.

**model and features**

- Model name, license (Apache-2.0 or MIT), parameter count (must be ≤ 8e9), and the weight or revision hash. Features are fields from the provided data or transforms of those fields.
- Define the judge score and the veto in terms of model outputs, not in terms of a country.
- Give `τ`, the comparison (≥ or >), and the assignment rule.

**other relevant information**

- Hardware, wall times, the exact command, the requirements pin, paths and hashes of the stacker and judge-score files, md5 of both TSVs, and the dry-run command.
- A short limitation: if the training file's labeled-country set changes, the dry-run fails and the previous md5 is no longer the expected output. That is intentional.
- Public and private scores are slices of this one TSV. There is no private-only patch.

Sentences that read as a rule violation even if a later code cleanup removes the offending call:

- "For France we added a hand-built gazetteer of communes and postal codes." Names the held-out country and an external database. Violates the open-set rule and the external-data ban.
- "We geocoded addresses with Nominatim / Google / a postal API to fill missing cities." A geocoder is named in the ban, regardless of country.
- "We one-hot encoded country with categories US and India and dropped other countries." Hard-codes the training pair and drops test entities.
- "After we saw the test file contained France, we special-cased `country == 'France'`." Hard-coded country name and use of test-file identity as the rule.
- "We swept the veto threshold against the public leaderboard and kept the value that raised the public score." The public leaderboard is one split of the same submitted file. Choosing `τ` from that score uses held-out feedback rather than the training labels, and it cannot be regenerated from the provided data alone. Kaggle does not universally ban this kind of search; Don't Overfit II is a known case of leaderboard probing. Under this contest's review and "regenerate from the provided data" rule, the sentence still looks like a violation. Do not rephrase it as "selected for countries with no training labels."
- "We hand-labeled a sample of the public test and used those labels to decide which vetoes to undo." Hand labeling of test or validation records is explicitly forbidden in the Kaggle rule text cited above, and it is not the training file.
- "The third country uses a French-specific model" or "we filtered the submission to US and India." Both restate a closed country set.

Sentences that match the rule:

- "Let C be the set of country strings on training rows that have a label. C is computed from the training file at runtime."
- "Where the entity's country string is not in C, the labeled-country judge is not applied. The decision is the baseline, except that a veto is undone when the model score on a pair the model already scored is at least τ. Each non-reference record is then assigned to at most one reference."
- "τ was selected on a validation split drawn from countries in C, before the stitch, and is not a function of country name. It was not selected on the public leaderboard."
- "The same rule applies to every country string absent from C. No country name is branched on. The model is [name], license [Apache-2.0 or MIT], [N] parameters, revision [hash]. No external gazetteer or geocoder was used."

If `τ` really was chosen by public-leaderboard search, or a gazetteer really was used, the methodology has to say that. The checklist item is disclosure (Pineau, NeurIPS), not a nicer paraphrase. Cleaning only the prose, while the code still calls a geocoder or branches on a country name, will not survive the review the brief says top teams receive.

### Gaps

- The Don't Overfit II discussion post was not opened; the probing description is from the Packt chapter that cites it.
- No organiser document found that uses the exact phrase "threshold fit on the public leaderboard" as an automatic disqualification. The violation judgment above is from this contest's regenerate-from-provided-data rule plus the one-file public/private split, not from a universal Kaggle ban.
- No public methodology template was found whose section titles are exactly methodology / blocking / model and features / other relevant information. Those headings are from the brief.

## Checklist for the zip

### Takeaway

One zip, one command, two TSV checksums, pinned model and judge-score hashes, a dry-run that fails when the labeled-country set changes, and a superset check so `candidate_pairs.tsv` contains every stitched match. The country set is recomputed from the training file inside that command.

### Cited Findings

- Kaggle requires a single archive with code, a serialized model, exact-version `requirements.txt` (or a Dockerfile with exact versions), `SETTINGS.json` as the only path source, `entry_points.md` with separate prepare / train / predict commands, `directory_structure.txt`, a README covering hardware, side effects, and assumptions, and code capable of generating the winning submission. Prediction is expected to load the saved model rather than retrain. — [Kaggle Winning Model Documentation Guidelines](https://www.kaggle.com/WinningModelDocumentationGuidelines); [ICR competition rules](https://www.kaggle.com/competitions/icr-identify-age-related-conditions/rules)
- DrivenData requires a fresh-system README, versioned dependencies, weights available without a retrain, and programmatic inference from raw competition data plus those weights, with no manual input. A notebook is acceptable only if it is that deterministic entry point. — [prize-winner-template](https://github.com/drivendataorg/prize-winner-template); [README_template.md](https://raw.githubusercontent.com/drivendataorg/prize-winner-template/main/README_template.md)
- Pineau v2.0 and the Papers with Code checklist both require dependency pins, training code, evaluation code, pre-trained models, and a README command that produces the reported result, plus the method used to choose hyperparameters. — [ReproducibilityChecklist.pdf](https://www.cs.mcgill.ca/~jpineau/ReproducibilityChecklist.pdf); [releasing-research-code](https://github.com/paperswithcode/releasing-research-code/blob/master/README.md)
- This contest additionally requires `output/matching_results.tsv`, `output/candidate_pairs.tsv` as the candidate set actually scored, regeneration of both from the provided data, the methodology sections named above, Apache/MIT models of at most 8B parameters, no external lookups, an open-set country rule, one file for both leaderboard splits, and an exact md5 against the upload-time stacker and the judge-score files used to build it. — Research brief (no public URL)

### Inferences

Checklist for the archive a reviewer can tick off. Paths match the brief.

- [ ] Single zip. Top level contains `README.md` (exact command, hardware, Python version, side effects, assumptions), pinned `requirements.txt` or an environment lock with exact versions, a settings file and no other hardcoded data paths, `entry_points.md` or the equivalent command block, code, the methodology document with the four sections, and `output/`.
- [ ] `output/matching_results.tsv` is the only scored file. The README states that public and private scores are splits of this file. There is no second private file and no country-specific patch file.
- [ ] `output/candidate_pairs.tsv` is the candidate set the model scored. A check in the entry command fails if any positive pair in `matching_results.tsv` is absent from `candidate_pairs.tsv`. Add-back pairs are included. Pairs that are not in the pinned judge-score file are not inserted as matches.
- [ ] Model pin: name, Apache-2.0 or MIT license, parameter count ≤ 8e9, and a revision or weight hash (sha256 of the weight file, or the upstream revision id plus a hash). The hash is in the README. The model is not an external knowledge base.
- [ ] Judge-score pin: relative path, sha256, row count, and which run it is (labeled-country stacker versus no-label scores). The README states these are the files that existed when the upload-time stacker was built. The stitch command verifies the hash before reading.
- [ ] Stacker pin: sha256 or md5 of the upload-time stacker artifact, and md5 of the regenerated `output/matching_results.tsv` and `output/candidate_pairs.tsv`. The command exits non-zero if regeneration does not match. Record the algorithm next to the digest so md5 and sha256 are not mixed.
- [ ] One command, copied verbatim in the README, rebuilds both TSVs from the provided data plus those pinned files. It recomputes `C` from training rows that have a label. It does not take a country name as an argument. Example shape, not a required filename: `python -m stitch --settings settings.json --check-md5`.
- [ ] Training code that produced the pinned scores is present and documented, separate from the stitch command, consistent with Kaggle's prepare/train/predict split and with DrivenData's inference-without-retrain practice. If a full retrain is too expensive for review, say so, and still make inference-plus-stitch rerunnable from weights and the provided data.
- [ ] Dry-run: with the real training file, the command prints `C` and passes the md5 check. Fixture A (new labeled country string) moves those ids onto `D_labeled`. Fixture B (labels removed) moves them onto `D_unlabeled`. Fixture C (unseen synthetic country) keeps the entity in the output exactly once under `D_unlabeled`. If recomputed `C` differs from the set logged at upload time, the command fails and does not write a file that claims the old md5.
- [ ] A text search of code, settings, README, and methodology finds no branch on a country literal and no geocoder or gazetteer import. Printing the computed members of `C` is allowed.
- [ ] Post-conditions scripted, not only described: required ids appear once; each non-reference id has at most one reference; country strings in the scored file are not a subset of `C`.
- [ ] No extra data directory of external lookups. Raw competition data may be omitted from the zip if the README says where the reviewer places it (DrivenData). Paths still come from the settings file (Kaggle).
- [ ] Methodology uses the "no training labels" wording in the previous section, including an honest sentence for how `τ` was chosen. Hardware, runtime, and seed (or a statement that the stitch is a pure function of the pinned scores and has no seed) are in "other relevant information."

Suggested command contract for the reviewer, as documentation rather than a mandated API:

```text
inputs:  train_labels, entities, required_ids,
         pinned stacker, pinned judge-score parquet(s), settings.json
step 1:  hash-check every pinned file
step 2:  C = countries on labeled training rows
step 3:  fail if C disagrees with the upload-time observation and --check-md5 is set
step 4:  for each required id, D_labeled if country in C else D_unlabeled
step 5:  enforce at most one reference per non-reference
step 6:  write output/matching_results.tsv and output/candidate_pairs.tsv
step 7:  fail if matches are not a subset of candidate pairs
step 8:  fail if md5 differs from the upload-time digests
```

`D_unlabeled` may only undo vetoes for pairs present in the judge-score parquet. Otherwise `candidate_pairs.tsv` would not be "the candidate set the model actually scored."

### Gaps

- No public sample of this contest's zip layout was found, so the checklist merges the brief with Kaggle's archive list and DrivenData's inference README. If the host's review form omits `directory_structure.txt` or `entry_points.md`, those two Kaggle items are still useful and are not a substitute for the files the brief names.
- The parameter-count check is a documentation requirement from the brief (≤ 8B). No source opened here gives a standard command for counting parameters; the README should say how the count was obtained (model card or a one-line parameter sum) so the reviewer can repeat it.
- Whether reviewers will retrain or only rerun inference is not stated for this contest. DrivenData says they rerun inference. The brief says the code regenerates both TSVs from the provided data. The checklist therefore requires a working stitch-from-pinned-scores command always, and a documented training command as well, without claiming the host will execute training.
