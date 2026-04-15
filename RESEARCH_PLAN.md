# Research Plan — Entity-Type Preservation in Clinical NLP Pipelines

> Single working document for this research direction. Consolidates prior audit, notebook fix, literature review, and research plan into one self-contained file. Uncommitted working files (`IMPROVEMENT_PROMPT.md`, `MASTER_PLAN.md`, `INVESTIGATION_NER_PIPELINE.md`) have been removed.

---

## 1. Thesis

Published clinical prediction pipelines that use biomedical NER (e.g. BC5CDR) routinely discard entity type at the extraction step, silently losing medication-vs-disease signal. This repo is an instance of a broader, unaudited pattern. A literature-only multi-repo audit can document the pattern, measure its prevalence in public MIMIC prediction projects, and propose a standard — without new experiments or MIMIC data access.

**Target venue**: ML4H, Clinical NLP Workshop (ACL), or AMIA Informatics Summit short paper.

---

## 2. The finding that motivated this (audit of this repo, 2026-04-15)

Three linked defects in this project's NER pipeline:

**2.1 Entity type stripped at extraction.** `notebooks/nlp_bc5cdr_ner.ipynb` cell 5 originally contained:
```python
unique_ner = set([str(ent).lower() for ent in nlp_model(text).ents])
```
`ent.label_` (`DISEASE` / `CHEMICAL`) is never captured. Both entity types are merged into a single untyped `word_list` per discharge note.

**2.2 "Disease embedding" is actually ICD-9, not NER.** `src/data/features.py:6-25` — `disease_embedding()` looks up `'IDX_' + code` where `code` comes from `src/data/reader.py:37-51::get_diseases()` → `diagnoses.csv` (structured ICD-9 codes). The 300-dim channel in the 390-dim input contract is **structured-only** and has no path from the NER notebook. The README's wording ("BC5CDR NER extracts drug/disease entities → 200-dim BioWordVec embeddings") misleads the reader into conflating these two separate paths.

**2.3 The only NER-consuming feature is untyped and disabled by default.** `src/data/features.py:28-40::get_wordvectors()` reads `k500_mean_wv` from `keyword_discharge_note_wv.pk` — a single 200-dim pooled vector per stay, with no entity-type distinction. It is activated only in `scripts/train_generator_wordvec.py` (which bumps `input_dim` 390 → 590); the default `scripts/train_generator.py` never uses NER output at all.

**Net effect**: zero paths from BC5CDR `CHEMICAL` labels to any medication-specific feature in the model. The `DISEASE` signal from NER is also collapsed into the same pooled vector as chemicals. The structured ICD-9 "disease embedding" is unrelated to the NER pipeline despite overlapping naming.

---

## 3. Source-side fix applied to this repo

**File changed**: `notebooks/nlp_bc5cdr_ner.ipynb` cells 5, 8, 14, 15.

| Cell | Change |
|------|--------|
| 5 | `get_matched()` now returns `{'all', 'disease', 'chemical'}` dict by filtering `ent.label_ == 'DISEASE'` / `'CHEMICAL'` |
| 8 | Unpacks the dict into `word_list` (backward-compat) + `disease_list` + `chemical_list` columns |
| 14 | Named-column selection (replaces `result_df.iloc[:, :5]` which silently dropped the new typed lists) |
| 15 | Adds `disease_mean_wv`, `disease_sum_wv`, `chemical_mean_wv`, `chemical_sum_wv` (200-dim each) to `keyword_discharge_note_wv.pk` |

**What this fix does not do**:
- Does not modify `src/data/features.py`, `src/data/generator.py`, or the 390-dim input contract.
- Does not retrain the model or regenerate the existing pickle — `keyword_discharge_note_wv.pk` on disk is stale until the notebook is re-run.
- Does not add `medication_embedding()`. Wiring typed embeddings into the 4-modality concat would break the 390-dim contract and the SHAP notebooks; that is out of scope for this research plan.

**Why not re-run**: cell 8's NER inference takes ~2h 19min on the reference machine and requires the MIMIC discharge-note pickles, which are not on this machine (`MIMIC_DATA_ROOT` is unset; `data/` contains only a README).

---

## 4. Literature context (April 2026 web search)

### 4.1 Prior art

| Reference | Contribution | Relevance to this plan |
|-----------|-------------|------------------------|
| ClinicalBERT (Huang et al. 2019, arxiv:1904.05342) | Transformer over MIMIC-III discharge summaries; 30-day readmission baseline | Does not use explicit NER feature engineering — full-text transformer bypasses the entity-type question |
| MuST / BioClinicalBERT (Scientific Reports 2023) | Multimodal graph-transformer (EHR + images + notes) | No NER feature engineering, no type ablation |
| SciBERT + structured (Westminster 2024) | BERT variant comparison for readmission; includes medication data | Includes medication features but does not measure typed-vs-untyped contribution |
| Predictive Modeling of Hospital Readmission (arxiv:2106.08488) | Data-table-level ablation (lab events least informative) | Ablation is at the TABLE level, not at the NER ENTITY-TYPE level |
| n2c2 2018 ADE shared task (JAMIA) | Rich drug entity types (DRUG, DOSAGE, ROUTE, ADE, …) on discharge notes | Framed as extraction benchmark, not as prediction feature utility |
| arxiv:2503.23050 (2025) | LLM characterization of notes + EHR fusion | Most recent; does not audit NER feature engineering |
| Grouped physiological + medication trends (PMC6474661) | MIMIC-II, 21 medication variables help ICU readmission | Closest prior evidence that medication signal matters — but uses structured medication variables, not NER-extracted entities |
| Unified Neural Architecture for Drug/Disease/Clinical NER (arxiv:1708.03447) | NER method paper | Method, not prediction |

### 4.2 Gaps

1. **No paper measures the predictive contribution of BC5CDR entity-type separation** on a clinical outcome task. BC5CDR is used as an NER benchmark, not audited as a feature source.
2. **No paper quantifies redundancy between ICD-9 structured disease features and NER-extracted disease features** when both are present in the same pipeline.
3. **No paper systematically audits published MIMIC prediction repos for entity-type preservation** as a methodological issue.

The third gap is the one this plan targets. It is the most defensible without new experiments.

---

## 5. Research plan — multi-repo audit

**Deliverable**: 5–10 page workshop paper. Literature-only + static code audit. **No new experiments, no MIMIC data required.**

### 5.1 Step 1 — candidate repo collection

- GitHub search via `gh api` / `gh search`: queries like `MIMIC readmission`, `MIMIC mortality NER`, `scispacy MIMIC`, `ClinicalBERT readmission`, `BioWordVec MIMIC prediction`.
- Filter to repos that (a) have source code, (b) run on MIMIC-III or MIMIC-IV, (c) use biomedical NER (scispacy, HuggingFace NER, MetaMap, cTAKES, …) as a feature source.
- Target: 15–25 repos, shortlisted from 50–100 candidates.
- **Pilot first**: start with 5 repos to verify the pattern exists before committing to a full 20-repo audit.

### 5.2 Step 2 — audit checklist per repo

| # | Question | Method |
|---|----------|--------|
| a | Which NER model is used? | Static search for `en_ner_*`, `scispacy`, `MetaMap`, `cTAKES`, `transformer` NER models |
| b | Does the extraction preserve `ent.label_` / equivalent? | grep for `.label_`, `.ent_type_`, `ent_label`, and check whether the label is stored or discarded |
| c | Does the pipeline pool all entities into one untyped vector? | Trace the flow from NER output to the model's input tensor |
| d | Does the same pipeline also use structured ICD-9/ICD-10 diagnoses? | grep for `ICD`, `diagnoses`, `D_ICD_DIAGNOSES` |
| e | Is there any redundancy or ablation analysis between structured and NER disease features? | README + paper search for "ablation", "redundant", "structured vs unstructured" |
| f | Does the README/paper claim "disease and drug" are used? Does the code use both? | Compare claim vs code |
| g | Downstream feature dimension and plug-in point | Find the `input_dim` / `concat` / `hstack` |

### 5.3 Step 3 — pattern classification

Cluster repos into categories:
- **P1**: type stripped at extraction (this repo's pattern)
- **P2**: type preserved but pooled late (entity types exist but are collapsed before the model)
- **P3**: fully typed (separate feature channels per entity type)
- **P4**: no NER at all (text → transformer, no explicit entity extraction)
- **P5**: NER extracted but unused (dead code)

### 5.4 Step 4 — frequency statistics + case study

- Report the distribution P1…P5 across the audited repos.
- Use this repo (MIMIC-III ICU readmission, LSTM-CNN) as the detailed P1 case study, including the concrete fix applied in §3.

### 5.5 Step 5 — recommendations

- A minimal "entity-type preservation checklist" for clinical NLP pipeline authors (6–8 items).
- Reference implementation: the diff applied to `notebooks/nlp_bc5cdr_ner.ipynb` in §3.

### 5.6 Risks & mitigation

| Risk | Mitigation |
|------|-----------|
| Pattern is rare (<20%): paper becomes n=1 case study | Run pilot on 5 repos first; if rare, pivot to a shorter "replication note" instead of a full audit paper |
| Pattern is universal (>80%): reviewers will ask "did you show the defect hurts performance?" | Include a "simulated ablation" section using this repo's existing stale pickle — compare the distribution of `k500_mean_wv` entries with/without typed splits — to suggest signal without retraining |
| GitHub repo quality variable | Define inclusion criteria upfront (must have source, must run NER, must connect NER to a prediction task) |
| Scope creep into MIMIC-IV / eICU | Stay MIMIC-III only for the audit; mention MIMIC-IV only in future-work |

---

## 6. Status

| Step | Status |
|------|--------|
| Audit this repo (§2) | done |
| Source-side fix to notebook (§3) | done (cells 5/8/14/15 edited, committed, not re-run) |
| Literature review (§4) | done |
| Candidate repo collection (§5.1, pilot set) | **done** — 5 repos selected (§9.1) |
| Audit checklist applied (§5.2) | in progress |
| Pattern classification (§5.3) | not started |
| Draft paper | not started |

**Next action**: apply the §5.2 checklist to pilot repo #2 (TimFrenzel/MIMIC-III-Clinical-NLP) and repo #3 (yzhouas/MIMIC-III_ICU_Readmission_Analysis).

---

## 7. Key references

- Huang et al. 2019. *ClinicalBERT: Modeling Clinical Notes and Predicting Hospital Readmission*. arxiv:1904.05342.
- *Prediction of 30-day hospital readmission with clinical notes and EHR information*. 2025. arxiv:2503.23050.
- *Predictive Modeling of Hospital Readmission: Challenges*. arxiv:2106.08488.
- *Predicting unplanned readmissions in the ICU: a multimodality evaluation*. Scientific Reports 2023. https://www.nature.com/articles/s41598-023-42372-y
- *Building Prediction Models for 30-Day Readmissions Using Structured + Unstructured Data*. PMC11271049.
- *Predicting ICU readmission using grouped physiological and medication trends*. PMC6474661.
- *Extracting adverse drug events from clinical notes: systematic review*. Journal of Biomedical Informatics 2024.
- *Ensemble of neural models for nested ADE and medication extraction (n2c2 2018)*. JAMIA. https://academic.oup.com/jamia/article/27/1/22/5518594
- *Unified Neural Architecture for Drug, Disease and Clinical Entity Recognition*. arxiv:1708.03447.
- *Using Medical Named Entity Recognition in Automatic ICD Prediction*. PMC12446077.

---

## 8. Notes

- **Data access**: `MIMIC_DATA_ROOT` is not set on this machine; `data/` contains only a README. Any empirical validation is blocked until data is available. This plan is designed to not require it.
- **Committed state**: notebook fix and this plan are committed on `main`. Commits: `fix: preserve BC5CDR entity type in NER notebook`, `docs: add research plan for entity-type preservation audit`. Not pushed.
- **Global rules**: no force-pushes, no co-author lines, no push without explicit user instruction.

---

## 9. Audit log

Accumulates concrete audit findings, per step. Append-only — do not delete old entries, update status in §6 instead.

### 9.1 Pilot repo selection (2026-04-15)

Searched GitHub via `gh api /search/repositories` with queries: `MIMIC+readmission` (30 results), `MIMIC+clinical+notes+NLP+readmission` (1), `ClinicalBERT+MIMIC` (~18), and related terms. Triaged ~50 candidates. Selected 5 for the pilot to maximise diversity of NER/NLP patterns.

**Pilot set**:

| # | Repo | ⭐ | Task | Hypothesised pattern | Why selected |
|---|------|----|------|----------------------|--------------|
| 1 | [JuhongPark/mimic-readmission-prediction](https://github.com/JuhongPark/mimic-readmission-prediction) | 0 | MIMIC-III 30-day ICU readmission | **P1** (type stripped at extraction — confirmed in §2) | Case study; the project that started this audit |
| 2 | [TimFrenzel/MIMIC-III-Clinical-NLP](https://github.com/TimFrenzel/MIMIC-III-Clinical-NLP) | 0 | MIMIC-III clinical NLP (stroke + critical conditions) | unknown — description says "spaCy, SciSpacy, MedSpacy, ClinicalBERT to extract entities" | Explicitly uses multiple NER stacks; highest prior likelihood of a typed-entity pattern |
| 3 | [yzhouas/MIMIC-III_ICU_Readmission_Analysis](https://github.com/yzhouas/MIMIC-III_ICU_Readmission_Analysis) | 27 | MIMIC-III ICU readmission (PLoS ONE paper) | unknown — predates current best practices | Lineage: this repo's README points to a fork (`JuhongPark/MIMIC-III_ICU_Readmission_Analysis`) of this codebase. Worth auditing the ancestor to see whether the pattern is inherited |
| 4 | [NikhilMY/ClinicalMind---Patient-Risk-Predictor](https://github.com/NikhilMY/ClinicalMind---Patient-Risk-Predictor) | 0 | MIMIC-III 30-day ICU readmission (ClinicalBERT + structured EHR + SHAP) | **P4** candidate (full-text transformer; entity types sidestepped) | Modern ClinicalBERT fusion with SHAP — tests the hypothesis that transformer-based pipelines avoid the problem entirely |
| 5 | [andrewwlong/mimic_bow](https://github.com/andrewwlong/mimic_bow) | 68 | MIMIC-III readmission from discharge summaries | **P4/P5** candidate (BOW — no NER at all) | Highest-star repo in the candidate pool; baseline for "traditional NLP without NER" |

**Rationale for mix**: one confirmed P1 (case study), one explicit scispacy/medspacy stack (highest NER exposure), one lineage ancestor, two "transformer or BOW" alternatives that should *not* exhibit the pattern. The mix is designed so that if the pattern is real, 1–3 should show it while 4–5 should not. If 4 or 5 unexpectedly show the pattern, that is an even stronger finding.

**Backlog** (not in pilot but retained for the full 15–25 repo audit if the pilot confirms the pattern):

| Repo | ⭐ | Rationale |
|------|----|-----------|
| [YaronBlinder/MIMIC-III_readmission](https://github.com/YaronBlinder/MIMIC-III_readmission) | 91 | Highest-star MIMIC-III readmission repo; description generic, needs a closer look |
| [apakbin/ICU72hReadmissionMIMICIII](https://github.com/apakbin/ICU72hReadmissionMIMICIII) | 28 | 72-hour variant, similar task |
| [Sue-Hi/NLP-MIMIC-III](https://github.com/Sue-Hi/NLP-MIMIC-III) | 12 | Explicit NLP focus |
| [mmrosek/MIMIC-ICU-Readmission-Prediction](https://github.com/mmrosek/MIMIC-ICU-Readmission-Prediction) | 7 | Notes + structured |
| [SashankBharadwaj11/ClinicalBert-Mimic3-Mortality-Readmission-Prediction](https://github.com/SashankBharadwaj11/ClinicalBert-Mimic3-Mortality-Readmission-Prediction) | 0 | Second ClinicalBERT variant for triangulation |
| [knaguib1/NLP-Hospital-Readmission-Prediction](https://github.com/knaguib1/NLP-Hospital-Readmission-Prediction) | 0 | ClinicalBERT on discharge notes |
| [lokesh9899/Mortality-Risk-Readmission-Prediction-NLP-Clinical-Bert-LLM](https://github.com/lokesh9899/Mortality-Risk-Readmission-Prediction-NLP-Clinical-Bert-LLM) | 2 | ClinicalBERT + XGBoost fusion |
| [altairBASIC/BDCC-reproduction](https://github.com/altairBASIC/BDCC-reproduction) | 1 | ICD prediction reproduction; not readmission but uses clinical NLP heavily |
| [Kbmukumbi/diabetes-nlp-structured-extraction](https://github.com/Kbmukumbi/diabetes-nlp-structured-extraction) | 0 | Rich medication entity types (DOSE/ROUTE/FREQUENCY) — potential "P3 fully typed" positive example |
