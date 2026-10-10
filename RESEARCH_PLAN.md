# Research Plan — Entity-Type Preservation in Clinical NLP Pipelines

> Single working document for this research direction. Consolidates prior audit, notebook fix, literature review, and research plan into one self-contained file. Uncommitted working files (`IMPROVEMENT_PROMPT.md`, `MASTER_PLAN.md`, `INVESTIGATION_NER_PIPELINE.md`) have been removed.

Notebook references below describe local exploratory analyses. The notebooks are excluded from version control.

---

## 1. Research goal

This repo is being developed into a research-focused investigation of **integrated clinical AI systems**: how specialist ML models (starting from the 30-day readmission LSTM-CNN this project began with) plug into broader hospital AI systems — LLM orchestrators, FHIR-native deployment, medication-safety agents, compliance frameworks.

**Two intertwined threads**:
1. **Diagnostic** (§2–§9): what is wrong with current research-grade clinical ML pipelines that blocks hospital integration? Source of evidence is the NER audit + multi-repo pilot.
2. **Prospective** (§10): where is the field moving, and what research angles exist to bridge the gap? Source of evidence is parallel literature surveys of systems-perspective research directions.

The primary short-term deliverable is the **parallel systems-perspective survey in §10**, with the diagnostic work (§2–§9) providing motivating evidence. Each §10 subsection is a standalone literature review + gap analysis + proposed angle that can be promoted to a full research thread, combined with another, or used as background for the primary paper.

**Target venues**: ML4H, Clinical NLP Workshop (ACL), AMIA Informatics Summit, npj Digital Medicine — venue selection depends on which survey direction is promoted to the primary contribution.

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
| Audit checklist applied (§5.2) | **done** — 5/5 (§9.1, §9.2, §9.3); verdict in §9.3 |
| Pattern classification (§5.3) | **done** — 5-category taxonomy (§9.2), finalized in §9.3 |
| **§10 Systems-perspective surveys (primary goal)** | **done** — 4 surveys + cross-cutting synthesis |
| §10.1 LLM clinical agents | **done** (drafted) |
| §10.2 Clinical NER medication safety | **done** (drafted, data-blocked) |
| §10.3 FHIR-native ML deployment | **done** (drafted) |
| §10.4 MLOps / FDA SaMD compliance | **done** (drafted; see §11.4 for revised thread recommendation) |
| §10.5 Cross-cutting synthesis | **done** |
| **§11 Hospital system × medication design focus** | **done** — 2-axis deep dive + integrated architecture |
| §11.1 Hospital system design deep dive | **done** (2026-04-21) |
| §11.2 Medication design deep dive | **done** (2026-04-21) |
| §11.3 Integrated architecture proposal | **done** (2026-04-21) |
| §11.4 Revised primary/secondary threads | **done** — supersedes §10.5 |
| **§12 Gap-driven supplementary investigation** | **done** — 13 gaps closed, revised action list supersedes §11.5 |
| §12.1 Hospital system supplementary | **done** (2026-04-21) — deployment, vendor, MIMIC→FHIR OSS, privacy, regulatory, server |
| §12.2 Medication design supplementary | **done** (2026-04-21) — anticoag, AMS, opioid/PDMP, TDM, geriatric, KB, adherence, PV |
| §12.3 Methodology & evaluation | **done** (2026-04-21) — UQ, calibration post-balance, DCA, trial design, drift |
| §12.4 Verification status table | **done** — 2 items fetched & cross-checked |
| §12.5 Revised follow-up actions (supersedes §11.5) | **done** |
| **§13 AI system design & management — MIT SDM lens** | **done** — re-framing §10-§12 as engineering-systems + management problem |
| §13.1–.2 Boundary, OPM, SVN | **done** (2026-04-21) |
| §13.3 QAS + ATAM tradeoffs | **done** |
| §13.4 Causal Loop Diagram | **done** — 4 loops (alert fatigue, trust erosion, regulatory ratchet, platform network effect) |
| §13.5 ADRs 001–008 | **done** — 7 decided, 1 open (MIMIC-III↔IV) |
| §13.6 DSM | **done** |
| §13.7 Real options | **done** — 4 options identified |
| §13.8 Dominant design / S-curve | **done** |
| §13.9 RACI + team topology | **done** |
| §13.10 Portfolio roadmap | **done** |
| §13.11 NIST AI RMF map | **done** |
| §13.12 TCO + unit economics | **done** |
| §13.13 Sociotechnical MLP | **done** |
| §13.14 Primary thread reframed → platform | **done** |
| §13.15 SDM action list (supersedes §12.5) | **done** |
| Draft paper | not started |

**Next action**: execute §13.15 action 1 — write ADR-001…008 as markdown under `docs/adr/`. This is the lightest-weight SDM artifact (pure documentation, no code), yields durable architectural decisions, and sets the scaffold that subsequent actions (MVIP spec, model card, CDS Hooks service stub) can reference.

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

### 9.2 Pilot audits: repos 2 and 3 (2026-04-15)

**Repo 2 — TimFrenzel/MIMIC-III-Clinical-NLP** (single `mimic_nlp.py`, 2,530 lines, 114 KB)

| Checklist | Finding |
|-----------|---------|
| NER model | scispacy + BC5CDR (`en_ner_bc5cdr_md`) + general-purpose spacy + MedSpacy + ClinicalBERT embeddings |
| Preserve `ent.label_`? | **Yes at extraction** — lines 354/358/376 store `(ent.text, ent.label_)` tuples for spacy, scispacy, and medspacy pipelines |
| Pooled untyped downstream? | **Yes at aggregation** — line 425–428 `set(ent[0].lower() for ent in row['scispacy_entities'])` drops the label for overlap statistics; line 464–466 `tokens = [token.lower() for token, _ in ents]` **explicitly unpacks and discards** the label when preparing Word2Vec training tokens |
| Uses structured ICD-9 alongside? | Filters notes BY ICD-9 stroke codes (430/431/434.x) as an input selector, but ICD-9 is not a model feature |
| Redundancy / ablation analysis | None — task is entity extraction + clustering, not prediction |
| Claim-vs-code consistency | Consistent. The README promises entity extraction; the code delivers. No prediction is claimed |
| Feature dimension / plug-in | Output is Word2Vec embeddings (100-d) and t-SNE visualisations; no prediction head |

**Pattern**: **P2 — type preserved at extraction, pooled untyped downstream.** The label is captured at extraction, rendered in HTML for visualisation (lines 631, 738), counted for entity-type distribution analysis (line 780), and **then explicitly discarded** in the Word2Vec token-preparation step (line 466: `for token, _ in ents`). Because the task is not prediction, no AUROC is harmed — but the aggregation pattern, if copied into a prediction pipeline, is exactly the failure mode this audit is about. Subtle but damning: a pipeline that installs BC5CDR *and* captures labels *and* visualises types still loses entity-type information when building its main output representation.

---

**Repo 3 — yzhouas/MIMIC-III_ICU_Readmission_Analysis** (Lin et al. 2018 bioRxiv paper code)

| Checklist | Finding |
|-----------|---------|
| NER model | **None.** No NLP imports, no spacy, no scispacy, no transformer text encoder. Pure LSTM on structured features |
| Preserve `ent.label_`? | N/A |
| Pooled untyped downstream? | N/A |
| Uses structured ICD-9 alongside? | **Yes.** `read_diagnose()` loads ICD-9 codes from `diagnoses.csv`; `get_diseases()` returns per-stay code lists; these feed a `get_embeddings()` lookup (likely a BioWordVec-by-code-string path) |
| Redundancy / ablation analysis | **Yes — explicit ablation directories** `mimic3models/readmission_no_d/`, `readmission_no_icd9/`, `readmission_f48/`. The authors clearly treated ICD-9 as a separable feature and ran ablations against it |
| Claim-vs-code consistency | Consistent — README and paper describe an LSTM on vitals + diagnoses + demographics, and that is what the code does. No NLP is claimed |
| Feature dimension / plug-in | LSTM input = timeseries + static features; exact dim not traced in this pass |

**Pattern**: **P4 — no NER at all.**

**Lineage finding (the important part)**: this is the **direct ancestor** of the audited repo. The functions `g_map`, `e_map`, `i_map`, `read_diagnose`, `get_diseases`, `read_demographic` in `mimic3models/readmission/main.py` are byte-for-byte identical (modulo Python 3 syntax updates and an `.ix` → `.loc` fix) to the audited repo's `config/defaults.py:1-21` and `src/data/reader.py:30-85`. The 300-dim "disease embedding" in the audited repo is a **direct inheritance** from yzhouas's ICD-9 code embedding — it has nothing to do with the NER notebook that was added later.

**Implication** (this changes the finding for the case study): the audited repo's P1 pattern is actually a *composite*:
1. The NER notebook (`nlp_bc5cdr_ner.ipynb`) was added on top of a non-NER yzhouas pipeline as a bolt-on.
2. At extraction, the notebook strips `ent.label_` — the classic P1.
3. Downstream, the NER pipeline is **never wired into the model's disease feature**; the model keeps using the inherited yzhouas ICD-9 path.
4. The README conflates "BC5CDR NER → 200-d BioWordVec" with "300-d disease embedding", misleading the reader into thinking they are the same data flow when they are two disjoint paths.

This is a more severe finding than originally thought. The source-side fix in commit `6ae97f8` addresses (2) but not (3) — even with typed embeddings now saved to `keyword_discharge_note_wv.pk`, the model's feature pipeline has no consumer for them.

---

**Revised draft taxonomy** (after 3 audits; will be finalized after repos 4 and 5):

| Pattern | Description | Example |
|---------|-------------|---------|
| P1 | Type stripped at NER extraction step | JuhongPark (original `get_matched` in cell 5 before fix) |
| P2 | Type preserved at extraction, then pooled untyped in downstream aggregation | TimFrenzel (`for token, _ in ents` at line 466) |
| P3 | Fully typed throughout the pipeline — each entity type is a separate feature channel | (none found yet) |
| P4 | No NER at all (baseline category — a pipeline cannot have the pattern if it never runs NER) | yzhouas |
| P5 | NER runs, but its outputs are orphaned from the model's feature pipeline; a "bolt-on" that is never connected | JuhongPark (after accounting for the yzhouas inheritance — NER notebook outputs do not reach `features.py::disease_embedding`, which uses structured ICD-9) |

This repo is thus a **P1 + P5 composite** — the worst of both worlds. TimFrenzel is a clean P2.

### 9.3 Pilot audits: repos 4 and 5 + NikhilMY disposition (2026-04-15)

**NikhilMY/ClinicalMind---Patient-Risk-Predictor** (original pilot pick #4): every source file in the repo is 0 bytes (`src/*.py`, `notebooks/*.ipynb`, `requirements.txt`, `README.md`). The impressive README description ("End-to-end clinical AI system predicting 30-day ICU readmission risk using ClinicalBERT + structured EHR fusion, SHAP explainability, interactive Plotly Dash dashboard") has no code behind it. **Classified as unauditable. Side finding to record in the paper: vaporware rate in the candidate query pool is non-zero.** Replaced with YaronBlinder/MIMIC-III_readmission (91 ⭐, highest-starred candidate).

**Repo 4 (replacement) — YaronBlinder/MIMIC-III_readmission** (Udacity ML Nanodegree capstone, 91 ⭐)

| Checklist | Finding |
|-----------|---------|
| NER model | **None.** Pipeline: PostgreSQL materialized view → pandas → feature engineering → XGBoost |
| Preserve `ent.label_`? | N/A |
| Pooled untyped? | N/A |
| Uses structured ICD-9? | No. Uses min/max/mean aggregates of labevents (urea_N, platelets, magnesium, albumin, calcium) and chartevents (RespRate, Glucose, HR, SysBP, DiasBP, Temp) via `all_data.sql`. No diagnoses, no notes |
| Ablation analysis | XGBoost feature importance only |
| Claim-vs-code | Consistent — README promises XGBoost on structured features, delivers that |
| Feature dim / plug-in | ~40 numerical features → XGBoost |

**Pattern**: **P4 — no NER at all.** The highest-starred MIMIC-III readmission repo in the candidate pool demonstrates that the dominant well-visible approach is structured-only, predating the NLP-fusion trend.

**Repo 5 — andrewwlong/mimic_bow** (68 ⭐)

| Checklist | Finding |
|-----------|---------|
| NER model | **None.** `CountVectorizer` (BOW) + `LogisticRegression(C=0.0001, penalty='l2')` |
| Preserve `ent.label_`? | N/A — no entity extraction |
| Pooled untyped? | All text collapsed to an untyped sparse term-frequency matrix |
| Uses structured ICD-9? | No — discharge summary text only |
| Ablation analysis | Hyperparameter tuning, no modality ablation |
| Claim-vs-code | Consistent with README ("BOW from discharge summaries") |
| Feature dim / plug-in | Sparse BOW → LogisticRegression |

**Pattern**: **P4 — no NER at all.** Second-highest-starred repo. Sanity-check bonus: swept `knaguib1/NLP-Hospital-Readmission-Prediction` (Spark-NLP for Healthcare, John Snow Labs) while replacing NikhilMY — uses `WordEmbeddingsModel.pretrained("embeddings_clinical")` but no `NerDLModel`/`NerConverter`. Word-level embeddings only, not entity extraction. Also **P4**.

**Pilot tally (5 repos)**:

| # | Repo | Pattern | Key evidence |
|---|------|---------|--------------|
| 1 | JuhongPark/mimic-readmission-prediction | **P1 + P5 composite** | Original cell 5 stripped `ent.label_`; 300-d "disease embedding" is inherited ICD-9 path from yzhouas, NER notebook orphaned |
| 2 | TimFrenzel/MIMIC-III-Clinical-NLP | **P2** | `(ent.text, ent.label_)` captured then `for token, _ in ents` discards label at word2vec training (line 466) |
| 3 | yzhouas/MIMIC-III_ICU_Readmission_Analysis | **P4** | No NER. **Direct ancestor of repo 1**, source of its structured-ICD-9 "disease embedding" |
| 4 | YaronBlinder/MIMIC-III_readmission | **P4** | Pure SQL → XGBoost on lab/chart/vital aggregates |
| 5 | andrewwlong/mimic_bow | **P4** | BOW + logistic regression on discharge summaries |

**Pilot verdict**: 3/5 are P4 (no NER). Of the 2/5 that do use biomedical NER, 100% (2/2) exhibit an entity-type preservation failure — one P1+P5 composite, one P2. Zero clean P3 examples found in the pilot.

**Implication for the paper framing**: the multi-repo audit thesis holds but needs narrower scope — "in repos that use biomedical NER as a feature source, entity-type preservation failures are endemic". The P4 repos are evidence that the field is substituting full-text transformers for explicit NER, so the pattern is more *historically* material than currently endemic. Pair the audit with the systems-perspective surveys (§10) to balance the "what went wrong in the NER-era" diagnosis with "where integrated clinical AI is heading".

---

## 10. Systems-perspective research surveys

Parallel literature surveys of research directions that move this repo's foundations (NER audit + LSTM-CNN readmission model + SHAP explainability + typed entity extraction post-fix) toward a hospital-deployable integrated AI system. Each subsection is a standalone literature review + gap analysis + proposed angle that connects to this repo's assets. Surveys can be promoted to full research threads, combined with each other, or used as background material for the primary paper.

**Survey slate**:
- §10.1 LLM clinical agents and multi-model orchestration
- §10.2 Clinical NER for medication safety and DDI alerting
- §10.3 FHIR-native ML deployment and CDSS standards
- §10.4 MLOps, GMLP, and the FDA SaMD compliance gap

Each survey follows the same template: **(a) space overview**, **(b) representative recent literature**, **(c) gap analysis**, **(d) connection to this repo's foundation**, **(e) candidate research angle**. Surveys are added in separate commits so that each can be tracked, refined, or rolled back independently.

### 10.1 LLM clinical agents and multi-model orchestration

**(a) Space overview.** A 2024–2026 wave of work treats the LLM not as the clinical predictor itself but as a **reasoner that orchestrates specialist ML components** (imaging models, genomics pipelines, EHR prediction heads, clinical NLP tools). Major vendors (Microsoft, OpenAI) and academic groups are building agent frameworks in which a general-purpose LLM decomposes a clinical question, calls the right specialist tool(s), and synthesises the results into a single clinician-facing view. The pattern is variously called "clinical copilot", "multi-agent CDSS", or "healthcare agent orchestrator".

**(b) Representative literature (April 2026 search)**:
- Microsoft's **Healthcare Agent Orchestrator** — Semantic Kernel-based multi-agent framework over imaging (DICOM), pathology (WSI), genomics, and structured EHR. Positioned as enterprise middleware for hospital AI integration.
- **ClinicalAgents** (arxiv:2603.26182): multi-agent clinical decision making with a dual-memory architecture.
- **"Enhancing Clinical Decision-Making: Integrating Multi-Agent Systems with Ethical AI Governance"** (arxiv:2504.03699v4) — argues that multi-agent CDSS needs an explicit governance layer.
- **"Multiagent AI Systems in Health Care: Envisioning Next-Generation Intelligence"** (PMC12360800).
- **"Multi-Agent LLM Orchestration Achieves Deterministic, High-Quality Decision Support"** (arxiv:2511.15755v2) — patterns for deterministic multi-agent execution, adaptable from incident-response to clinical.
- **Penda Health + OpenAI clinical copilot** (2025) — production EHR-embedded real-time safety net running during every visit.
- **"Integrating LLMs for enhanced predictive analytics in healthcare"** — npj Digital Medicine, 2026.
- **"Developing next-generation cancer care management with multi-agent orchestration"** (Microsoft, May 2025) — modular multimodal agents over imaging, pathology, genomics, and notes.
- **"LLM Agents for Biomedicine: A Comprehensive Review"** (MDPI Information 2025).
- **AgenticHealthAI/Awesome-AI-Agents-for-Healthcare** (GitHub awesome list).

**(c) Gap analysis**:
1. **Legacy-model integration is under-explored.** Current orchestrators are built around modern foundation-model tools (CLIP-style image encoders, genomics-GPT, EHR-BERT). Legacy specialist models — a 2018 LSTM-CNN trained on MIMIC-III, for example — are not first-class tools, even though real hospitals have years of such models in production.
2. **Tool-interface standards are missing.** Each orchestrator declares its own function-calling format. There is no consensus on how a prediction tool should expose (i) input schema, (ii) prediction + calibrated uncertainty, (iii) SHAP / XAI output, (iv) fairness flags, (v) provenance (model version, training data window). MCP (Model Context Protocol) is emerging as a general answer but no clinical-specific profile exists.
3. **Explainability hand-off is ad-hoc.** Most multi-agent CDSS papers discuss "explainable results aligned with clinical expectations" but do not specify how the LLM consumes and renders per-prediction SHAP / counterfactual / uncertainty outputs from a specialist tool.
4. **Regulatory framing is thin.** When an LLM orchestrator invokes an FDA-cleared SaMD as a tool, is the whole pipeline newly in scope for SaMD review? Unanswered in the literature as of April 2026.

**(d) Connection to this repo**:
- The existing LSTM-CNN + SHAP pipeline is exactly the kind of legacy specialist ML that gap (1) flags as under-served. It has a defined input schema, a checkpoint (once data is available), and SHAP output — everything an orchestrator would need to call it as a tool.
- The typed-entity NER output (post-fix, §3) is another candidate tool — a "medication entity extraction" microservice the orchestrator can call on a new note, returning `disease_list` and `chemical_list` plus their 200-d embeddings.
- The multi-repo audit (§9) showed that most research-era pipelines have unstable / under-documented interfaces → reinforces gap (2) from primary evidence.

**(e) Candidate research angle**:

> **"Wrapping legacy clinical ML models as tools for LLM orchestrators: an interface specification with a MIMIC-III readmission case study"**

A short paper or position piece that:
1. Proposes a minimal tool-interface specification for a clinical prediction microservice (FHIR-subset input → prediction + calibrated uncertainty + SHAP attribution + fairness flag + model version + training-data window).
2. Demonstrates the spec by wrapping this repo's LSTM-CNN as a tool an LLM agent can call.
3. Shows a small end-to-end example: an LLM agent, given a synthetic patient summary, calls (i) the readmission tool and (ii) the typed NER tool, then renders an integrated risk narrative with grounded citations to the tool outputs.
4. Discusses the SaMD-scope question raised in gap (4).

Deliverable: position paper + minimal reference code. No MIMIC data required if the example uses synthetic/toy input. Venue: ML4H system demo track, or npj Digital Medicine short report.

### 10.2 Clinical NER for medication safety and DDI alerting

**(a) Space overview.** Clinical NER for medication extraction sits at the intersection of information-extraction research and clinical decision support (CDSS). The canonical task is adverse drug event (ADE) extraction from discharge summaries with rich entity types (DRUG, DOSAGE, ROUTE, FREQUENCY, DURATION, REASON, ADE). The **n2c2 2018 ADE & Medication Extraction shared task** put 505 MIMIC-III discharge summaries at the centre of this line of work. In parallel, operational CDSS research has documented a severe **alert-fatigue** problem: >90% of drug-drug interaction (DDI) alerts in deployed systems are overridden, and >50% of overrides are clinically inappropriate. The two literatures — extraction quality and alert utility — rarely cross.

**(b) Representative literature (April 2026 search)**:
- **n2c2 2018 ADE & Medication Extraction shared task** — 505 MIMIC-III discharge summaries with rich entity typing (Medication, Strength, Dosage, Duration, Frequency, Form, Route, Reason, ADE).
- **"ADE and medication relation extraction in EHRs with ensemble deep learning"** — JAMIA 27(1), 2020 (`academic.oup.com/jamia/article/27/1/39/5544735`).
- **"Extracting adverse drug events from clinical notes: a systematic review"** — JBI 151, 2024 (`sciencedirect.com/science/article/pii/S1532046424000212`) — 76 studies total, 8 LLM-based approaches, trend toward LLM-first pipelines.
- **MultiADE** — multi-domain ADE benchmark (arxiv:2405.18015, 2024).
- **"Knowledge-aware neural attentive joint modeling for drug-safety entity/relation extraction"** — JMIR Medical Informatics 2020.
- **"Optimizing DDI Alerts Using a Multidimensional Approach"** — PMC6398362.
- **"Improving the specificity of DDI alerts: can it be done?"** — PMC9218784.
- **"Effect of electronic DDI alerts on patient and clinician outcomes: systematic review"** — PMC12451929. Key finding: *"no current evidence of a clinically important effect of DDI alerts on patient outcomes"*.
- **"Overriding DDI alerts in CDSS: scoping review"** — AHRQ PSNet / ResearchGate 361144614.

**(c) Gap analysis**:
1. **Extraction quality and alert utility are studied separately.** The NLP community optimises F1 on ADE extraction benchmarks; the CDSS community measures override rates on legacy rule-based alerts. Nobody has connected "better-structured extraction → smarter alert prioritisation".
2. **Patient-specific risk context is absent from alerts.** Current DDI alerts fire on drug combinations regardless of the individual patient's risk. A readmission or mortality risk score combined with a specific drug pair could in principle yield a personalised risk score; this has not been operationalised.
3. **Typed entity outputs are not used at alert time.** Extraction pipelines produce rich typed outputs (DRUG, DOSAGE, ROUTE, REASON), but alert systems still match on drug-name strings only. The typing is discarded before reaching the alert rule engine — exactly the same methodological pattern this repo's multi-repo audit (§9) documents at training time.
4. **Override-aware prioritisation is missing.** No published work trains an alert suppression model on override patterns using NLP-extracted medications as features.

**(d) Connection to this repo**:
- The typed BC5CDR output from the §3 fix provides exactly what gap (3) needs: `disease_list` + `chemical_list` columns with per-entity 200-d embeddings, preserving the type distinction that legacy pipelines discard at the NER→feature boundary.
- The LSTM-CNN readmission risk score is exactly what gap (2) needs: a patient-specific risk variable that can condition alert prioritisation.
- The methodological parallel with §9 is strong — the failure mode this paper would document (type-agnostic alert matching) is the *runtime* mirror of the *training-time* failure mode §9 documents.

**(e) Candidate research angle**:

> **"Patient-risk-conditioned DDI alert prioritisation from typed biomedical NER: a MIMIC-III retrospective simulation"**

Proof-of-concept paper that:
1. Uses typed BC5CDR output (from §3's fix) to extract medications from MIMIC-III discharge notes.
2. For each (patient, drug-pair) that would trigger a DDI alert in a reference ruleset (DrugBank, RxNorm DDI), scores the patient's readmission risk with this repo's LSTM-CNN.
3. Re-ranks alerts by `(rule severity × patient risk)` and compares the top-K under a plausible alert-capacity budget (e.g. 10 alerts per shift) against the unranked baseline.
4. Reports how many of the known-inappropriate overrides from the literature are suppressed under the new ranking.

Deliverable: short paper + reference implementation. **This angle is not data-free** — the retrospective simulation requires MIMIC-III access. Flag for "next step once `MIMIC_DATA_ROOT` is set". Venue: ML4H, AMIA Symposium short paper, JAMIA Open.

### 10.3 FHIR-native ML deployment and CDSS standards

**(a) Space overview.** FHIR (Fast Healthcare Interoperability Resources) is the dominant standard for health data exchange. **FHIR R5** (2024) added explicit support for AI/ML scenarios; CMS/ONC incentivise hospital adoption with Medicare reimbursement-linked requirements. A growing body of work packages ML predictions as FHIR `RiskAssessment` resources so they can be consumed by any FHIR-compliant EHR. But the practical gap between an academic ML model and a deployable FHIR microservice remains 6–12 months of data-wrangling and interface work per deployment.

**(b) Representative literature (April 2026 search)**:
- **"ML-Enabled Clinical Information Systems Using FHIR Standards: Scoping Review"** — PMC10468818 — 39 articles reviewed: 18 CDSS, 10 data mgmt/analytics, 11 APIs. Key finding: *"many intelligent systems lacked EHR interoperability and externally validated evidence of clinical efficacy."*
- **"ML-Enhanced Architecture Model for Integrated and FHIR-Based Health Data"** — MDPI Information 16(12), 1054.
- **FHIR-Former** (PMC12646377) — enhances clinical predictions by combining FHIR with LLMs.
- **"Development of an Interoperable and Easily Transferable CDSS Deployment Platform"** — JMIR 2022 (`jmir.org/2022/7/e37928`).
- **"State-of-the-Art FHIR-based Data Model and Structure Implementations: Systematic Scoping Review"** — PMC11472501.
- **"Experience in Developing an FHIR Medical Data Management Platform"** — PMC6981801.
- **"Using CDSS to bring predictive models to the glaucoma clinic"** — PMC7854795 — rare concrete deployment case study.
- **2024 Year in Review: FHIR Milestones** — Itirra blog (`itirra.com/blog/2024-year-in-review-key-milestones-for-fhir-and-the-rise-of-itirra-a-new-force-in-health-tech/`).

**(c) Gap analysis**:
1. **No reference implementation for a MIMIC-III research model as a FHIR microservice.** Thousands of MIMIC-based ML papers exist; none publish an adapter that takes FHIR `Observation` + `Condition` + `MedicationRequest` inputs and returns a `RiskAssessment` output.
2. **SHAP-to-FHIR mapping is unspecified.** There is no agreed resource for per-prediction attribution / feature importance. Papers that expose SHAP do so in ad-hoc JSON payloads, not a standard resource or extension.
3. **Model versioning and provenance have no standard home.** FHIR resources have `meta.versionId` but ML model versions are usually tracked separately. The audit trail a `RiskAssessment` should embed (model version, training data window, calibration date, drift-monitoring status) has no conventional field.
4. **Fairness annotations have no FHIR home.** If a model flags reduced confidence for a demographic subgroup, there is no standard place to put that flag in the output resource — a growing problem as FDA GMLP guidance (§10.4) increases transparency requirements.

**(d) Connection to this repo**:
- This repo's 4 modalities map almost directly to FHIR resources: timeseries → `Observation` (vitals/labs), ICD-9 → `Condition`, demographics → `Patient`, discharge notes → `DocumentReference`. A FHIR input adapter is a well-defined piece of work.
- The LSTM-CNN's output + SHAP values + MC Dropout uncertainty (if added) form the natural payload for a `RiskAssessment` resource with custom extensions.
- The multi-repo audit (§9) demonstrated that most research repos have ad-hoc feature formats and undocumented interfaces. A FHIR adapter is the concrete answer to "how do we make this deployable?".

**(e) Candidate research angle**:

> **"MIMIC-to-FHIR: a reference implementation for deploying research ML models as FHIR-native microservices"**

A systems / engineering paper that:
1. Defines a minimum FHIR profile for clinical prediction microservices — required input resources, `RiskAssessment` output extensions for SHAP attribution, model version, training-data window, fairness flags.
2. Provides a reference implementation that wraps this repo's LSTM-CNN behind a FHIR API (FastAPI + `fhir.resources`).
3. Validates the adapter on **Synthea / SyntheticMass** synthetic FHIR data (publicly available, no credentialed access required).
4. Publishes the profile as a FHIR ImplementationGuide that other research groups can extend.

Deliverable: engineering paper + reference code. **No MIMIC-III data required** — Synthea covers the validation scenario. Venue: JAMIA, JMIR Medical Informatics, HIMSS / AMIA systems track.

### 10.4 MLOps, GMLP, and the FDA SaMD compliance gap

**(a) Space overview.** The FDA's **AI/ML SaMD (Software as a Medical Device) Action Plan** — proposed 2021, finalized December 2024 — established a Total Product Life Cycle (TPLC) regulatory approach for AI/ML medical devices. In January 2025 FDA published *"AI-Enabled Device Software Functions: Lifecycle Management and Marketing Submission Recommendations"*, making lifecycle expectations explicit for marketing submissions. The accompanying **Good Machine Learning Practice (GMLP)** guidance lists 10 guiding principles. **Predetermined Change Control Plans (PCCPs)** allow manufacturers to pre-specify permissible model modifications without re-submission. Transparency — public submission summaries, training-data descriptions, known failure modes — is required. This regulatory framework applies to any clinical ML model intended for deployment, yet the academic ML literature rarely addresses compliance head-on.

**(b) Representative literature (April 2026 search)**:
- **FDA** — *Artificial Intelligence in Software as a Medical Device* landing page (`fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-software-medical-device`).
- **FDA** — *Good Machine Learning Practice for Medical Device Development: Guiding Principles* — the 10 GMLP principles.
- **"ML-Enabled Medical Devices Authorized by the US FDA in 2024: Regulatory Characteristics, Predicate Lineage, and Transparency Reporting"** — PMC12730494.
- **"US FDA Regulation of Clinical Software in the Era of AI and ML"** — Mayo Clinic Proceedings: Digital Health, 2025 (`mcpdigitalhealth.org/article/S2949-7612(25)00038-0/fulltext`).
- **"FDA Expectations for AI/ML Model Training in SaMD (2025 guide)"** — rookqs.com practitioner guide.
- **"A Complete Guide to the FDA's AI/ML Guidance for Medical Devices"** — ketryx.com.
- **"How the FDA Reviews AI and Machine Learning Medical Devices: Complete 2025 Guide"** — complizen.ai.
- Academic overview: PMC12264609 — "US FDA Regulation of Clinical Software in the Era of AI/ML".

**(c) Gap analysis**:
1. **GMLP compliance of published research repos is unmeasured.** No study systematically audits public MIMIC prediction repositories against the 10 GMLP principles. The §9 audit of 5 repos in this plan strongly suggests most would fail even basic items (training-data window undocumented, no bias audit, no versioning, no change-control plan, no monitoring).
2. **PCCPs are never specified in research papers.** Research models are frozen artefacts — retrained ad hoc, without a documented permitted-change envelope. A "research-to-deployment readiness score" per repo could quantify this gap.
3. **Transparency reporting is ad-hoc.** Published papers include different subsets of the FDA-required transparency information (intended use, architecture, training data, subgroup performance). A checklist audit would reveal how much is routinely missing.
4. **Academic incentives misalign with GMLP.** Academic metrics (AUROC on a benchmark split) reward discrimination; GMLP rewards *safety* (documented failure modes, fairness, monitoring, change control). No bridging framework currently exists to translate academic models into GMLP-compliant artifacts.

**(d) Connection to this repo**:
- This repo is a concrete case study for gap (1): the §9 multi-repo audit methodology applies directly but with a *different* checklist (the 10 GMLP principles instead of the 5-pattern entity-type taxonomy). Same audit infrastructure, new scoring dimension.
- The P1+P5 pattern this repo exhibits maps to multiple GMLP violations: Principle 3 (training data representative of intended patient population — unverified), Principle 8 (performance of the human-AI team — absent, no deployment), Principle 10 (monitoring in deployment — absent).
- The FHIR adapter from §10.3 is a prerequisite for pursuing SaMD clearance of this specific model — §10.3 and §10.4 interlock as an "engineering + compliance" pair.

**(e) Candidate research angle**:

> **"GMLP compliance of public MIMIC prediction repositories: a retrospective audit against the FDA's 10 guiding principles"**

An extension of the §9 multi-repo audit methodology with a different checklist:
1. Curate 20–30 public MIMIC prediction repositories (readmission, mortality, sepsis, LOS) — reuse the §9 candidate pool plus targeted searches.
2. Score each repo against the 10 GMLP principles on a tri-state scale (absent / partial / complete), using README, code, and paper evidence.
3. Report per-principle compliance rates and aggregate a "deployment-readiness score" distribution across repos.
4. Propose a one-page "GMLP minimum" checklist that research repos can adopt as a README supplement to close the largest gaps cheaply.

Deliverable: audit paper + GMLP minimum checklist template. **Purely literature + code-audit based; no data required.** Strongly complementary with §9 — same method, different checklist; shares candidate-repo pool. Venue: npj Digital Medicine, JAMIA, ML4H methodology track.

### 10.5 Cross-cutting synthesis and next-step decision

**How the four surveys interlock**:
- §10.1 (LLM orchestration) and §10.3 (FHIR-native microservices) specify *how* a legacy research model plugs into a modern clinical AI stack. §10.3 is the concrete pipe; §10.1 is the consumer.
- §10.2 (medication safety) and §10.1 (orchestration) describe a *specific use case*: the LLM orchestrator calls the readmission tool AND the typed-NER tool AND a DDI ruleset to produce a patient-personalised alert.
- §10.4 (GMLP compliance) and §9 (entity-type audit) share the methodology: both are static audits of public repos. §10.4 extends the §9 audit pool and method to a new checklist.
- §10.3 (FHIR) and §10.4 (GMLP) are the "engineering + compliance" pair: §10.3 is what you build; §10.4 is the checklist it must satisfy.

**Data-free vs data-required**:

| Survey | Data required? |
|--------|---------------|
| §10.1 LLM orchestrator | Synthetic only — **data-free** |
| §10.2 Medication safety | **Requires MIMIC-III** for the retrospective simulation |
| §10.3 FHIR microservice | Synthea synthetic FHIR — **data-free** |
| §10.4 GMLP audit | Static code audit — **data-free** |

Three of the four candidate threads can proceed without `MIMIC_DATA_ROOT`. Only §10.2 is blocked on data access.

**Promotion-to-primary decision matrix**:

| Criterion | §10.1 | §10.2 | §10.3 | §10.4 |
|-----------|-------|-------|-------|-------|
| Novelty of contribution | high (legacy-wrap) | high (risk × typed NER) | medium (MIMIC→FHIR ref impl) | medium-high (GMLP audit) |
| Data-free | ✓ | ✗ | ✓ | ✓ |
| Uses §3 fix directly | indirect | ✓✓ direct | indirect | no |
| Uses §9 audit method directly | no | indirect | no | ✓✓ direct |
| Paper size | short | short-medium | medium | medium |
| Venue visibility | high | medium | medium | high |

**Recommended primary thread**: **§10.4 (GMLP compliance audit)**. Rationale: (i) strictly data-free, (ii) directly extends §9 audit methodology already in place, (iii) maps to an explicit external regulatory framework (FDA GMLP) that provides a principled checklist instead of an invented one, (iv) reviewers cannot easily object to the absence of empirical performance results because the deliverable is a compliance audit, not a prediction claim, (v) audit-style work is well-received at npj Digital Medicine / JAMIA / ML4H methodology.

**Recommended secondary thread**: **§10.1 (legacy-model wrapping for LLM orchestrators)** — a position paper plus minimal reference implementation using this repo's LSTM-CNN as the concrete case study; complementary to §10.4.

**Deferred**: **§10.2** (needs MIMIC-III for the retrospective). The §3 NER fix is the precondition already in place; the experimental half waits on data access.

**Possibly merge**: **§10.3** can be folded into §10.1 as the interface specification the position paper proposes, reducing the number of independent threads from four to two plus one deferred.

**Next concrete action**: pilot §10.4 by extending the §9 candidate-repo pool (add 10–15 MIMIC mortality / sepsis / LOS repos) and dry-run the GMLP checklist on 3 of them to verify the scoring rubric is tractable. If the pilot shows clean score separation, promote §10.4 to full execution.

---

## 11. Hospital system design × medication design focus (2026-04-21)

A focused research direction targeting the two axes that most directly translate this repo's assets (LSTM-CNN readmission predictor + typed BC5CDR NER after the §3 fix + SHAP explainability) into deployable hospital artifacts: **(i) hospital system integration design** and **(ii) medication decision-support design**. This section deepens §10.1 (LLM orchestration) + §10.3 (FHIR deployment) on the system side and §10.2 (DDI alerting) on the medication side with 2024–2026 literature not previously cited, then proposes a unified integration architecture and revises the primary-thread recommendation (§11.4 supersedes §10.5).

### 11.1 Hospital system design — deep dive

**(a) The 2026 integration stack.** Four interlocking standards now define production-grade hospital AI integration. (1) **SMART on FHIR** (OAuth2 + FHIR API + app launch). 100 % of ONC-certified EHRs expose SMART endpoints; 90–95 % of U.S. hospitals reported a functional SMART endpoint by 2023. (2) **CDS Hooks** (JSON CDS-service invocations at workflow events — `patient-view`, `order-select`, `order-sign`, `medication-prescribe`). A cluster-randomized trial (Dziadkowiec et al. *JAMIA* 2022, PMC9382378) showed CDS Hooks prompts *increase* SMART on FHIR app utilization — the two are complements, not competitors. (3) **FHIR versions**. Epic and Oracle Health (Cerner) both require FHIR R4 + SMART on FHIR 2.1 (DSTU2 deprecated by Cerner Dec 2025); industry uptake of R5 is still low but the **R5 Subscription Backport IG** lets R4 systems consume R5 topic-based subscriptions without a full version migration; R6 is expected late 2026 with most clinical/admin resources going normative. (4) **Model Context Protocol (MCP)** and its healthcare extension **HMCP** (Innovaccer, 2025) — the emerging standard for LLM tool access to FHIR data and clinical workflows. Open-source MCP–FHIR servers (WSO2, AgentCare, Momentum, the arXiv:2506.13800 reference framework) all appeared in 2025.

**(b) Representative literature (2024–2026)**:
- Dziadkowiec et al. *Using CDS Hooks to increase SMART on FHIR app utilization: a cluster-randomized trial*, JAMIA 2022 (PMC9382378).
- *AI Integration in Clinical Decision Support Systems Enhancing Patient Outcomes through SMART on FHIR and CDS Hooks* (ResearchGate 384042934, 2024).
- *Development of SMART on FHIR Interoperable CDS for ED Patients with Pneumonia and Pilot Deployment into Novel Epic EHR Environments* (AHRQ Digital Healthcare Research, 2024) — rare concrete multi-vendor deployment case.
- *Enhancing Clinical Decision Support and EHR Insights through LLMs and the Model Context Protocol* (arXiv:2506.13800, 2025) — open-source MCP+FHIR+LLM agent framework.
- *Artificial Intelligence Platform Architecture for Hospital Systems: Systematic Review*, JMIR 2025 e79788 (PMC12710730) — 5-layer model (infrastructure / data / algorithm / application / security-compliance) synthesizing HIMSS digital-health, WHO-HIS, sociotechnical, NIST BDRA frameworks.
- *Designing a System Integration Architecture for an ML-Based CDSS in the Emergency Department* (ScienceDirect 2025, S1062737525000484).
- *Designing Clinical Decision Support Systems (CDSS) — Review*, JMIR 2025 e63733 — data quality + temporal dependency + dynamic risk flagged as systemic CDSS failure modes.
- *Privacy-Preserving Clinical Decision Support for Emergency Triage Using LLMs* (MDPI 2076-3417/15/15/8412, 2025).
- HL7 — *R5 Subscriptions Backport IG v1.2.0-ballot* (`build.fhir.org/ig/HL7/fhir-subscription-backport-ig/`).
- Innovaccer — *Healthcare Model Context Protocol (HMCP)* spec (`github.com/innovaccer/Healthcare-MCP`).
- FDA & EMA — *Guiding Principles of Good AI Practice in Drug Development*, Jan 14 2026 — 10 joint principles across nonclinical / clinical / post-market / manufacturing. Complements FDA GMLP (§10.4) on the *drug* side.

**(c) Gap analysis (hospital systems)**:
1. **No reference implementation for a research-grade MIMIC model (e.g. a 2018-era LSTM-CNN) wrapped as a CDS Hooks service.** Published CDS Hooks examples use rule-based or greenfield-ML logic; legacy ML is unrepresented.
2. **No published HMCP / MCP tool manifest for a readmission-risk model.** The tool-interface gap §10.1 flagged is still open with no clinical-ML reference.
3. **5-layer CDSS architectures assume new ML.** The JMIR 2025 systematic review (PMC12710730) describes the blueprint but ships no legacy-ML integration example — template, not reference implementation.
4. **Epic / Oracle vendor integration is still bespoke.** Even with SMART on FHIR 2.1 + CDS Hooks, each vendor's auth, App Orchard / Oracle Health marketplace, and UI embedding path differs. "Standards-only" deployment narratives understate the vendor-specific work.
5. **Audit trail across the stack is not standardized.** Per-prediction provenance (model version, training-data window, calibration epoch, drift monitor status, fairness flag) has no standard FHIR home — mirrors §10.3 gaps but now across CDS Hooks responses *and* MCP tool results.

**(d) Connection to this repo**:
- The LSTM-CNN + SHAP + typed NER stack maps cleanly to a CDS Hooks `patient-view` service (readmission risk card at chart open) and a `medication-prescribe` service (risk-conditioned alert).
- 4 modalities → FHIR R4 resources: timeseries → `Observation`; ICD-9 → `Condition`; demographics → `Patient`; discharge note → `DocumentReference`.
- The typed BC5CDR output from the §3 fix is the input for a `medication-prescribe` hook's CHEMICAL channel; DISEASE channel cross-checks `Condition`.
- The multi-repo audit (§9) documented ad-hoc feature interfaces that make deployment bespoke per repo — motivates a standardized reference implementation.

**(e) Candidate research angle (hospital systems)**:

> **"A reference integration architecture for legacy clinical ML: wrapping a MIMIC-trained LSTM-CNN as a CDS Hooks service, a SMART on FHIR app, and an HMCP tool"**

Engineering + position paper that:
1. Defines a *minimum viable integration profile* (MVIP) covering (i) SMART on FHIR launch context, (ii) CDS Hooks service manifest + card format for risk prediction + SHAP attribution, (iii) HMCP tool manifest for MCP-based LLM orchestrators, (iv) `RiskAssessment` output FHIR profile with standard extensions for model version, training-data window, calibration epoch, fairness flag, uncertainty.
2. Publishes a reference implementation wrapping this repo's LSTM-CNN (weights optional; spec can be demonstrated with a dummy checkpoint).
3. Validates on Synthea / SyntheticMass synthetic FHIR data (no MIMIC-III required).
4. Discusses the regulatory implications — at what point does the wrapped legacy model become in-scope for FDA SaMD review?

Deliverables: spec + reference repo + position paper. Venue: JAMIA / JMIR Medical Informatics / ML4H systems demo track.

### 11.2 Medication design — deep dive

**(a) Space overview — the medication-safety stack.** Medication-related harms and readmissions are a first-class hospital problem. A 2021 systematic review (Uitvlugt et al., *Front. Pharmacol.* PMC8077030) pegs **drug-related readmissions at ~9 % pooled prevalence** among adult hospital admissions, with **15.4–22.2 % classified as preventable**; in one cohort study, **16 % of readmissions were medication-related and 40 % of those preventable**, decomposed as 35 % prescribing errors + 35 % non-adherence + 30 % transition errors. High-risk drug classes: anticoagulants, antibiotics, psychotropics, chemotherapy. Polypharmacy (≥ 5 meds) is a dominant risk modifier. This sits at the center of several parallel medication-CDSS tracks — CPOE, DDI alerting, medication reconciliation, pharmacogenomics (PGx), and closed-loop administration — each of which has matured independently and which no prior work unifies under a single outcome-risk-conditioned prioritization.

**(b) Representative literature (2024–2026)**:

*CPOE & ML-augmented prescribing*:
- *Implementation of CPOE Systems: Impacts, Barriers, Facilitators — Review*, 2024 (PMC11536519).
- *International transferability of ML models for detecting medication errors in general internal medicine* (2025) — generalization is limited across sites.

*DDI alert fatigue mitigation*:
- *The potential for leveraging ML to filter medication alerts* (PMC9006688) — LightGBM achieves 0.192 precision at 0.99 sensitivity → **54.1 % alert-volume reduction**.
- *ML-based DDI prediction: critical review of models, limitations, data challenges*, Front. Pharmacol. 2025 (PMC12344460).
- *AI-powered DDI prediction, vulnerable populations, regulatory insights*, Front. Pharmacol. 2025 (PMC12380558).
- *Closing the loop: human-augmented, mechanistically enhanced AI for proactive DDI management*, Front. Pharmacol. 2026 (fphar.2026.1767646).
- *A comprehensive landscape of AI applications in broad-spectrum drug interaction prediction: systematic review of 147 studies (2018–2024)*, J. Cheminformatics 2025 (s13321-025-01093-2).
- *ML for predicting drug-drug interactions: Graph neural networks and beyond*, 2025 (S2452310025000113).

*Medication reconciliation*:
- *Analytical approaches for medication reconciliation-related topics: a scoping review* (medRxiv 2025.03.16.25324060) — first scoping review of analytics methods in med reconciliation.
- RCT: *Effectiveness of digital platform in reducing unintentional medication discrepancies* — 18.8 % → 3.4 % (PMC12219133, 2025).
- *Medication Reconciliation at Transition of Care in a Geriatric Primary Care Setting* (PubMed 40296246, 2025) — pharmacist-led: 5.5 discrepancies / patient; 30.3 % require pre-visit intervention.

*Pharmacogenomics CDSS*:
- Caudle et al. *Advancing Clinical Pharmacogenomics Worldwide Through CPIC*, *Clin. Pharmacol. Ther.* 2025 (PubMed 40678821). 28 active guidelines, 34 genes, 164 drugs, 128 institutions, 40 commercial labs; CPIC API ~80 K monthly queries.
- Hinderer et al. *A Pharmacogenomics CDS Service Based on FHIR and CDS Hooks* (PubMed 30605914) — canonical FHIR+CDS-Hooks PGx pattern.
- Brown et al. *Impact of PGx on P&T Committee Formulary Management*, *JACCP* 2025 (jac5.2057).
- Epic foundational genomics module + CDS Hooks delivery of CPIC content.

*Closed-loop medication administration*:
- Sorensen et al. *Evaluating the Impact of Smart Infusion Pump Interoperability on Reducing Medication Administration Errors: Systematic Literature Review*, 2025 (PMC12009041) — 87.9 % U.S. hospitals have smart pumps; only **9–15 % have pump-EHR interoperability**; autoprogramming: high-alert drug errors 84 % → 16 %; keystroke reduction 86 %.
- *Maximizing the impact of smart-pump-EHR interoperability in critical care*, 2024 (S0964339724001940).
- *Closed-Loop Medication Management with EHR in U.S. and Finnish Hospitals* (PMC10488169).
- ISMP Guideline 6.6 — bidirectional SMART pump-EHR interoperability.

*Drug-related readmission epidemiology (cross-cutting)*:
- Uitvlugt et al. *Medication-Related Hospital Readmissions Within 30 Days of Discharge* (PMC8077030, 2021).
- *Prevalence of and Risk Factors for Drug-Related Readmissions in Older Adults: Systematic Review & Meta-Analysis* (PMC10770220, 2023) — 9 % pooled prevalence.
- *Potentially inappropriate medications at discharge: prevalence, predictors, association with early readmission and ED visits in older adults* (PMC12342278, 2025).

*Regulatory*:
- FDA / EMA *Guiding Principles of Good AI Practice in Drug Development*, Jan 14 2026 — 10 principles across nonclinical / clinical / post-market / manufacturing; complements FDA GMLP (§10.4) on the *drug* rather than *device* side.

**(c) Gap analysis (medication design)**:
1. **ML-DDI filtering is not personalized by outcome risk.** LightGBM reduces alert volume ~54 % at 0.99 sensitivity by learning *override patterns* — but the patient's own readmission / mortality / LOS risk is not part of the prioritization feature set. An alert clinically inconsequential for a low-risk patient may be actionable for a high-risk patient; current filters do not distinguish.
2. **Medication reconciliation tools are not prioritized by readmission risk.** A pharmacist with a fixed queue capacity and 5.5 discrepancies / patient would benefit from a risk-ordered queue; no published tool incorporates an LSTM / CNN risk score.
3. **PGx CDSS does not cross-condition with outcome models.** CPIC guidelines specify gene/drug actionability but do not fire differentially by patient outcome risk — a high-readmission-risk patient on a CPIC-level-A drug warrants a stronger alert than a low-risk patient on the same drug.
4. **Closed-loop medication administration has a 75 %+ implementation gap.** 87.9 % of U.S. hospitals have smart pumps but only 9–15 % have bidirectional pump-EHR interoperability. This is the single largest deployment gap in the medication-safety stack and the natural "last mile" for CDSS actions.
5. **Readmission ML systems treat readmission as a unitary outcome.** 9–16 % of readmissions are medication-related and 15–40 % of those preventable — a causally decomposable subcategory — yet readmission-prediction papers (this repo included) predict a single binary target and do not model the medication-mediated sub-outcome separately. A medication-mediated readmission sub-head is a concrete modeling extension.

**(d) Connection to this repo**:
- The LSTM-CNN readmission score is the conditioning variable gaps (1–3) need — a patient-level outcome risk available at the moment of DDI alert firing, reconciliation queuing, or PGx lookup.
- The typed BC5CDR output (post-§3 fix) is the medication-entity feature gap (1) needs for patient-specific DDI prioritization — it turns a drug-pair match against a static ruleset into a drug-pair match *weighted by per-patient extraction quality and typed coverage*.
- The SHAP attribution is the explanation layer gap (3) needs — when alerting a high-risk patient on a CPIC-actionable drug, SHAP tells the clinician *why* this patient is high-risk in a way that complements the CPIC recommendation.
- The §9 lineage finding (ICD-9 disease path inherited from yzhouas, NER orphaned) maps cleanly onto a medication-design gap: the NER CHEMICAL channel has never been wired to an alert pathway in this repo; §11.2 identifies exactly where that wiring should go.

**(e) Candidate research angle (medication design)**:

> **"Outcome-risk-conditioned medication safety: a unified prioritization framework for DDI alerts, reconciliation queues, and PGx recommendations, with a MIMIC-III retrospective simulation"**

Extends §10.2 across the full medication-safety stack rather than DDI alone:
1. Define a generalized prioritization function `priority(i) = f(static_severity_i, outcome_risk, medication_risk_class, PGx_actionability)` applicable to DDI alerts, reconciliation discrepancies, and PGx recommendations — a *single* prioritization model rather than three separate alert rankers.
2. Use the LSTM-CNN readmission score as the `outcome_risk` signal; use the typed BC5CDR output for drug/disease grounding; use CPIC annotations for PGx actionability.
3. Retrospectively simulate on MIMIC-III:
   - DDI alerts: match typed CHEMICAL list against DrugBank / RxNorm DDI ruleset; compute priority.
   - Reconciliation queue: simulate discharge-day queue re-ranking.
   - PGx: simulate CPIC-level-A drug exposure × outcome risk.
4. Report: alert-budget capacity curves (10 / shift, 20 / shift, …); percentage of known-inappropriate overrides (from literature) suppressed; re-identified medication-mediated readmission cases.
5. **Data-required**: needs `MIMIC_DATA_ROOT`. Blocked until data access; §3 typed NER is the precondition already in place.

Deliverable: paper + reference implementation. Venue: ML4H, AMIA Informatics Summit, JAMIA Open.

### 11.3 Integrated architecture — hospital system × medication design

The two axes couple through the *medication-mediated readmission causal pathway*: 9–16 % of all readmissions are drug-related and 15–40 % of those preventable. A hospital AI stack that (i) predicts readmission risk (system side) and (ii) intervenes in medication safety (medication side) but does not couple them leaves this causal fraction un-targeted. A stack that couples them — the readmission score conditions the medication-safety triage — targets the preventable fraction explicitly.

**Proposed reference architecture** (mapped to the 5-layer CDSS model from §11.1):

| Layer | Component | Role |
|:------|:----------|:-----|
| 1 Infrastructure | FHIR R4 server (Epic / Oracle / HAPI) + R5 Subscription Backport | Data substrate + real-time event stream |
| 2 Data | `Patient`, `Observation`, `Condition`, `MedicationRequest`, `MedicationAdministration`, `DocumentReference` | EHR native |
| 3 Algorithm | (a) LSTM-CNN readmission predictor (this repo); (b) typed BC5CDR NER microservice (this repo + §3 fix); (c) LightGBM DDI-alert filter; (d) CPIC PGx lookup; (e) drift / calibration monitors | Intelligence |
| 4 Application | CDS Hooks services: `patient-view` → risk card; `medication-prescribe` → outcome-risk-conditioned DDI + PGx alert; SMART on FHIR dashboard; HMCP tool endpoints consumed by an LLM clinical copilot | Workflow integration |
| 5 Security / compliance | SMART auth (OAuth2 + SMART scopes); per-prediction audit on `RiskAssessment`; GMLP (§10.4) for the predictor; FDA / EMA 2026 drug-AI principles for any AI-assisted medication decisions | Governance |

The LSTM-CNN readmission score is the *cross-layer bridging signal*: it lives in layer 3 (prediction), returns via layer 4 (`patient-view` card), re-enters layer 3 as a conditioning input to the DDI filter and PGx triage, then returns again via a second layer-4 surface (`medication-prescribe` hook). Closed-loop execution extends into medication administration where smart-pump interoperability exists (the 9–15 % of hospitals in §11.2(b)).

### 11.4 Primary & secondary thread recommendations (supersedes §10.5)

| Thread | Data-free? | Uses §3 fix? | Uses §9 method? | Paper size | Priority |
|:-------|:-----------|:-------------|:----------------|:-----------|:---------|
| **§11.1(e)** Reference integration architecture | ✓ Synthea | indirect | no | medium | **primary** |
| §11.2(e) Outcome-risk-conditioned medication safety | ✗ MIMIC | ✓ direct | no | short-medium | deferred (data-blocked) |
| §10.4 GMLP compliance audit | ✓ | no | ✓ direct | medium | **secondary** |
| §11.3 Integrated architecture position | ✓ | ✓ indirect | indirect | short | **secondary** |

**Recommended primary**: **§11.1(e) — reference integration architecture**.
Rationale: (i) data-free via Synthea; (ii) directly exercises the two assets in this repo that the existing §10 plan flagged as under-used (the LSTM-CNN + the post-fix typed NER); (iii) fills a concrete gap (no published HMCP tool manifest for a legacy clinical ML model) with a minimal-footprint artifact reviewers can replicate; (iv) complementary with §10.4 — the GMLP audit becomes much sharper when a concrete reference architecture exists as the compliance target; (v) shorter to produce than a full retrospective simulation and not blocked on MIMIC-III access.

**Recommended secondary combo**:
1. **§10.4 GMLP compliance audit** as the methodology paper (static audit, different checklist, shared candidate pool with §9).
2. **§11.3 integrated-architecture position piece** as the vision / roadmap paper tying the two axes together.

**Deferred**: **§11.2(e) outcome-risk-conditioned medication safety** — requires MIMIC-III access.

### 11.5 Next concrete actions

1. **CDS Hooks service stub** — scaffold a FastAPI service exposing `/cds-services` + `patient-view` that accepts a FHIR Prefetch bundle and returns a Card with a dummy readmission prediction + SHAP top-5 features. No real model weights required for the interface paper.
2. **FHIR input adapter** — write a mapping from the repo's 4 modalities to FHIR R4 resources and a small Synthea-to-model input converter. This is the concrete deliverable the §10.3 FHIR angle stopped at "to be written."
3. **HMCP tool manifest** — draft a JSON manifest describing the readmission tool and the typed-NER tool for consumption by an MCP-speaking LLM orchestrator. Compare against the Innovaccer HMCP spec + the arXiv:2506.13800 reference framework.
4. **Minimum `RiskAssessment` extension draft** — propose FHIR extensions for (model version, training-data window, calibration epoch, fairness flag, uncertainty, SHAP attribution JSON) and write it up as a mini ImplementationGuide.
5. **In parallel**: proceed with the §10.4 GMLP pilot — extend the §9 candidate pool by 10–15 MIMIC repos (mortality / sepsis / LOS) and dry-run the GMLP scoring rubric on 3 of them.

---

## 12. Gap-driven supplementary investigation (2026-04-21, continued)

§11 identified two axes and four research threads but the supporting literature left clear unsurveyed gaps. §12 systematically catalogues those gaps, adds the 2024–2026 literature that closes them, verifies selected load-bearing claims, and revises the follow-up action list for the primary and secondary threads. §12.5 supersedes §11.5.

### 12.0 Gap map

| # | Short name | Source gap | Why it matters | Evidence class |
|:--|:-----------|:-----------|:---------------|:---------------|
| A | Real-world deployment cases | §11.1 cited standards, no metric-backed deployment | Shows whether §11.1(e) is reinventing solved problems | Case study |
| B | Vendor-specific paths | §11.1(c)-4 flagged but unenumerated | MVIP must target the paths Epic / Oracle actually accept | Developer docs |
| C | MIMIC→FHIR tooling | §11.5 action 2 was "to be written" | Reuse vs rebuild decision for the primary thread | Existing OSS |
| D | Regulatory breadth | §10.4 = FDA only | US-state + EU + international compliance matrix | Law / guidance |
| E | High-risk drug classes | §11.2 named 4 classes with no class-specific CDS review | Test cases for §11.2(e) | Domain CDS literature |
| F | Drug KB consolidation | §11.2 used DrugBank / RxNorm / CPIC loosely | MedicationRequest interop requires canonical coding | Terminology |
| G | Uncertainty quantification | §10.3 + §11.1(e) assume "uncertainty extension" with no method choice | MVIP must name a method | Method literature |
| H | Calibration after balanced sampling | §10.3 flagged recalibration without method survey | This repo *uses* balanced sampling — directly load-bearing | 2026 evidence |
| I | Privacy-preserving ML | §11 mentioned briefly | Many hospital deployments require FL / DP | Method + deployment |
| J | Clinical utility measurement | §10 + §11 used AUROC only | DCA / net benefit is standard-of-care in 2025 | Methodology |
| K | Prospective trial design | absent from §10 / §11 | A deployed model needs a prospective evaluation plan | Trial design |
| L | Drift monitoring | §10.4 principle 10 only partial | MVIP needs a concrete drift-flag | Monitoring |
| M | Pharmacovigilance integration | absent | Readmission signal can feed AEMS / FAERS | Regulatory infra |

### 12.1 Hospital system design — supplementary findings (gaps A, B, C, D, I)

**12.1.1 Real-world deployment evidence (A).**
- **Penda Health + OpenAI AI Consult** (arXiv:2507.16947, July 2025). Jan–Apr 2025 study, 15 primary-care clinics in Nairobi, 39,849 patient visits (20,859 with AI Consult, 18,990 without). **16 % relative reduction in diagnostic errors; 13 % reduction in treatment errors.** No patient-reported-outcome difference in the study window (underpowered for that endpoint). Zero AI-caused harm. Clinician survey: 100 % reported quality improvement, 75 % "substantial". Unresolved critical "red" alerts dropped from ~40 % → ~20 % after focused training. LLM copilot runs in background during visits; de-identified documentation sent to OpenAI API at key workflow points. A PATH-partnered RCT is now underway. *Verification (WebFetch on arXiv abstract)*: metrics confirmed; specific workflow-invocation pattern + PHI-handling architecture + task decomposition are reserved for the full paper and cannot be cited at that detail level yet.
- **AIM ECG-AI** (PubMed 40276542, 2025) — multicenter pragmatic cluster-randomized trial of EHR-integrated ECG-AI CDSS for low-LVEF detection.
- **VTE prophylaxis in TBI** (Implementation Science s13012-024-01386-4, 2024) — pragmatic stepped-wedge hybrid type II trial across 9 heterogeneous hospitals with external control sites.
- **ePNa pneumonia CDSS** (Intermountain Healthcare → AJRCCM stepped-wedge, plus a 2025 SMART-on-FHIR port for multi-EHR validation) — direct precedent for the §11.1(e) legacy-model wrapping approach.

Implication: Penda demonstrates a production LLM-first copilot pattern. §11.1(e) — wrapping a legacy specialist ML model as a CDS Hooks + SMART + HMCP triple — is complementary, not duplicative.

**12.1.2 Vendor-specific integration paths (B).**
- **Epic**: App Orchard has been re-branded **Epic Showroom**. 2025 developer program adds "Validating AI Features" + ED Pre-Arrival + Specialty Registries playbooks. Epic reports ~200 AI features in development. Integration is FHIR R4 + SMART on FHIR 2.1; client ID / secret, redirect URIs, SMART scopes, launch contexts are all part of the Showroom registration.
- **Oracle Health (Cerner)**: Cerner Code Console → developer platform. FHIR R4 + OAuth 2.0 for PowerChart. DSTU2 deprecated Dec 2025.
- Both vendors converge on **FHIR R4 + SMART on FHIR 2.1 + CDS Hooks** — these are the MVIP targets for §11.1(e).

**12.1.3 MIMIC → FHIR existing tooling (C).** *Critical finding — rewrites §11.5 action 2.*
- **kind-lab/mimic-fhir v2.1.0** (2024-10-28, *WebFetch-verified*). FHIR R4. MIMIC-IV and MIMIC-IV-ED. PostgreSQL SQL scripts + HAPI FHIR JPA Server (Maven-run, no Docker stack). **No ML-specific extensions** — SHAP, model version, uncertainty, drift flag are not represented. MIT-Lab maintained, active development (501 commits main).
- **srdc/mimic-iv-to-fhir** — uses the toFHIR transformation engine.
- **skfit-uni-luebeck/mimic4fhir** — MIMIC-IV → HL7 FHIR R4 *or* German Medical Informatics Initiative core datasets.
- **MIMIC IG v1.3.0** at `mimic.mit.edu/fhir/` — 25 profiles + 2 extensions + 35 ValueSets + 34 CodeSystems (JAMIA 2023, academic.oup.com/jamia/article/30/4/718/6998091).
- **MIMIC-IV Clinical Database Demo on FHIR v2.1.0** on PhysioNet — 100 patients / ~915 K resources (demo); 315 K patients / ~5.84 M resources (full).
- **Open MIMIC-III gap**: all four projects target MIMIC-IV. This repo uses MIMIC-III. Follow-up is either a MIMIC-III → MIMIC-IV schema mapping (most MIMIC-III tables have direct MIMIC-IV equivalents) or a narrow custom MIMIC-III adapter.

**12.1.4 Privacy-preserving ML in hospital deployments (I).**
- Differential privacy with calibrated noise on gradients: scalable, practical; incurs accuracy cost that shrinks with optimal ε budget.
- Homomorphic encryption and secure MPC: stronger privacy, higher compute cost — less deployment-friendly than DP.
- **PRIFLIC-Net** — DP Gaussian noise + HE for local updates in multi-hospital FL.
- Raj et al. (Wiley *Concurrency & Computation* 2025, cpe.70365) — FL framework with *clinician feedback* loop for multi-hospital CDS.
- Breast-cancer FL-DP (PMC12003885) and e-health FL frameworks 2025.

**12.1.5 Regulatory matrix 2026 (D).** *Extends §10.4 beyond FDA.*

| Jurisdiction | Framework | Effective | Applies to this repo? |
|:-------------|:----------|:----------|:----------------------|
| US FDA | GMLP (10 principles) + final PCCP guidance (Aug 2025) | Active. 30 devices (~10.2 %) cleared with PCCP in 2025; 6 PCCP example scenarios published | Yes if deployed clinically |
| US FDA | Jan 2025 draft *Considerations for Use of AI to Support Regulatory Decision-Making for Drug and Biological Products* (7-step risk-based credibility assessment) | Draft | Indirect (model as drug-evidence) |
| US FDA + EMA | Jan 14, 2026 *Guiding Principles of Good AI Practice in Drug Development* (10 joint principles) | Active (non-binding) | Indirect |
| US state — CO | SB24-205 (comprehensive AI Act covering healthcare as a "consequential decision" domain for high-risk AI) | Pushed to 2026-06-30 | **Yes** — readmission scoring at deployer tier |
| US state — CO | SB21-169 (insurance) | Active since 2021 | Only if deployed in insurance CDS |
| US state — CA | CCRC FEHA ADS regulations | 2025-10-01 | Employment-ADS only |
| EU | AI Act: general-purpose obligations | 2025-05 | If deployed in EU |
| EU | AI Act: MDR class IIa/IIb/III + IVDR A-D = high-risk medical AI | 2027-08-02 (36 months after entry into force; majority of AI Act in force 2026-08-02) | **Yes** if EU deployment |
| EU | MDR / IVDR | Active (AI Act allows consolidation with MDR technical docs) | Yes |
| FDA | FAERS → **AEMS** transition (adverse event monitoring) | 2026-03-11 | Indirect |

**12.1.6 FHIR server choice (C continued).** HAPI FHIR (open-source Java, research-friendly, fastest Docker setup) → recommended default for the §11.1(e) reference implementation. Aidbox (dev-focused, GraphQL + SQL + subscriptions + free dev license, Kubernetes-native) → better for a production pilot. Firely Server (.NET, FHIR-creators-backed) → compliance-strong commercial. LinuxForHealth (IBM) → open-source alternative.

### 12.2 Medication design — supplementary findings (gaps E, F, M)

**12.2.1 Anticoagulation CDS (E, highest-risk class).**
- Warfarin ML on MIMIC-III: *Application of ML techniques for warfarin dosage prediction: a case study on the MIMIC-III dataset* (PMC11784795 / PeerJ cs-2612). **Direct dataset overlap with this repo.**
- Reinforcement-learning warfarin dosing: 80.8 % excellent-responder ratio vs. 41.6 % standard practice; 99.5 % safety-responder ratio vs. 83.1 %.
- CURATE.AI retrospective for warfarin dose optimisation (PMC12079355).
- Bleeding-risk ML in AF anticoagulation (PubMed 40517766, 2025).
- ML for warfarin therapy — systematic review (MDPI *Pharmaceuticals* 18/10/1544, 2025).
- Narrative review on AI / ML for precision warfarin dosing (EJCP 2026, 00228-026-03997-w).
- **CYP2C9 / VKORC1 genotypes explain < 50 % of dose variability** → strong argument for outcome-risk conditioning (§11.2(e)) — PGx alone is insufficient; a patient-level risk score adds complementary signal.
- **Implication for §11.2(e)**: warfarin × MIMIC-III × readmission risk × CYP2C9/VKORC1 is the cleanest first test case — there is a prior-work baseline to beat, it is the top-risk class, and the data overlap is exact.

**12.2.2 Antimicrobial stewardship CDS (E).**
- KINBIOTICS (JMIR Human Factors 2025, e66699) — AI CDSS for sepsis antibiotic therapy, use-case analysis.
- OneChoice / OneChoice Fusion — AI CDSSs for bacteremia management evaluated via clinical vignettes with specialist physicians.
- Sepsis prediction ML — 30–60 % mortality reduction in quasi-experimental studies, 58 % in RCTs; AUROC 0.68–0.99 across models.
- WGS + clinical metadata ML — > 90 % accuracy for resistance markers (hours vs. days turnaround).
- **Critical systemic finding** ("*Bug Wars*", PMC12345688, and JAC advance article dkaf470): *"AI-CDSS have largely failed to integrate into clinical settings due to narrow scope of defined medical tasks, failure to consider stakeholder workflow, and lack of proper evaluation frameworks."* — direct primary-evidence for the §11.1(e) and §11.4 motivation that interface + workflow + evaluation frameworks are the binding constraint, not model accuracy.

**12.2.3 Opioid stewardship + PDMP (E).**
- PDMP-EHR integration → **~3× increase** in PDMP queries; ~⅓ of prescribers query via EHR (up from ⅕ in 2019). ONC + AHRQ framework.
- ML for identifying potentially-inappropriate opioid prescribing (PMC9307080) — logistic regression, LASSO, CART, RF, GBM.
- NIC RxGov — ML-powered PDMP platform.
- Implication: another medication class that benefits from outcome-risk conditioning (§11.2(e) generalization pathway).

**12.2.4 Precision dosing / TDM (E).**
- **OPTIVAN** (JMIR 2025 e63983) — 7-feature ML (age, BMI, glucose, BUN, eGFR, Hct, daily dose / BW) predicting whether AUC₂₄/MIC is in therapeutic range for vancomycin.
- ML-based PK model selection + averaging outperforms single PK models (CPT:PSP 2025 psp4.70084).
- Continuous-learning MIPD reduces prediction error up to 13 % vs. static population models.
- Bayesian forecasting now recommended in guidelines for vancomycin + beta-lactams.
- Connection: "dose recommendation" is another medication-safety output that §11.2(e) prioritization can condition on outcome risk.

**12.2.5 Geriatric / polypharmacy safety (E).**
- **STOPP / START v3** (2023, Eur Geriatr Med, PMC10447584) — 190 criteria (v2: 114).
- Two multicenter trials of CDSS for STOPP / START v2 → **15 % and 62 % advice uptake** — *suboptimal*. Implementation barriers: time constraints, high patient volume, notification fatigue.
- Connection: exactly the alert-fatigue problem §11.2(e) addresses. Outcome-risk-conditioned prioritization of STOPP / START advice is a direct test application.

**12.2.6 Drug knowledge-base consolidation (F).**
- **RxNorm is the integration hub**: First DataBank, Multum, Micromedex, Gold Standard, SNOMED CT drug component, MeSH, ATC, DrugBank all map into RxNorm.
- **NDC** → 11-digit HIPAA format via RxNorm.
- **UMLS Metathesaurus** integrates SNOMED CT + LOINC + RxNorm.
- **DrugBank powers RxNorm's DDI API.**
- March 2025: RxClass removed "Therapeutic Role" classes (SNOMED inactivated grouper).
- **Implication for this repo**: the typed BC5CDR CHEMICAL output from §3 should be *normalised to RxNorm* (not DrugBank directly) for interoperability with EHR `MedicationRequest`, `MedicationAdministration`, and RxNorm DDI API. A small RxNorm-normalisation layer becomes a concrete intermediate artifact (§12.5 action 6).

**12.2.7 Medication adherence ML (E).**
- **NYU Langone HF cohort — 34,697 patients** (JAMIA Dec 2025, ocaf162) — ML prediction of adherence to guideline-directed HF therapy with pharmacy-fill + neighborhood-level linkage.
- Random Forest **AUC 0.92–0.98** for post-discharge compliance (medRxiv 2025.09.14.25335382).
- **MedAdhereAI** (medRxiv 2025.07.01.25330675) — interpretable LR (AUC 0.82) + RF (AUC 0.77).
- Connection: §11.2(a) decomposes drug-related readmission causes as 35 % prescribing / 35 % **non-adherence** / 30 % transition. Adherence is *the single largest causal sub-category* §11.2(e) does not yet touch. Pairing an adherence-prediction head with the readmission head is a natural multi-head extension.

**12.2.8 Pharmacovigilance AI (M).**
- **FDA "Elsa"** (launched June 2025) — agency-wide internal LLM for adverse-event summary, label comparison, code generation.
- FDA Jan 2025 draft *Considerations for the Use of AI to Support Regulatory Decision-Making for Drug and Biological Products* — 7-step risk-based credibility assessment framework.
- **FAERS data migrated to AEMS** on 2026-03-11.
- Implication: a hospital-deployed readmission predictor with a documented medication-mediated-readmission sub-head (per §11.2(a) decomposition) becomes a first-class pharmacovigilance signal source. Upstream research angle: auto-generate AEMS-ready structured signals from deployed CDSS events.

### 12.3 Methodology & evaluation — supplementary findings (gaps G, H, J, K, L)

**12.3.1 Uncertainty quantification for LSTM-CNN (G).**
- **MC Dropout** — simple retrofit, randomly deactivates nodes at inference; *Expected Calibration Error can be unstable* (e.g., ECE 0.7588 in a ChestX-ray14 pilot before remediation).
- **Deep Ensembles** — 9-member ensemble in the same pilot achieved SOTA AUROC 0.8559 with stable calibration. Higher compute cost but strong production default.
- **Conformal Prediction** — distribution-free, provides per-prediction sets with user-specified error-rate guarantee. Can wrap any base model without retraining.
- **Recommendation for §11.1(e) MVIP**: expose uncertainty as a Conformal-Prediction interval (distribution-free guarantee); fall back to Deep Ensembles when a parametric mean ± sd interpretation is required.

**12.3.2 Calibration after balanced sampling (H). CRITICAL finding — directly affects this repo.**
- **medRxiv 2026.03.04.26347634** — *"Class imbalance correction in AI models leads to miscalibrated clinical predictions: a real-world evaluation."* Class-imbalance mitigation *artificially alters the pre-test probability distribution seen during training*, recasting the population context to which risk estimates apply → miscalibration that Platt-scaling does not fully repair.
- **arxiv:2410.18144** — *Using Platt's scaling for calibration after undersampling — limitations*. Platt-alone is insufficient when training was on an undersampled majority class.
- Isotonic regression delivers consistent probability-reliability gains in many settings, but **can collapse AUC to 0.51–0.56 in heterogeneous real-world data** — context-dependent, not a safe default.
- **SFI-aware calibration** (Scientific Reports 2025, s41598-025-32656-w) — hybrid framework with label-free deployment; calibration parameters developed on labelled data but applied without labels on new sites.
- **Implication for this repo**: §11.1(e) MVIP must *name* the calibration strategy explicitly. This repo's balanced sampling (README "Key decisions") means any deployment requires more than Platt scaling. The 2026 medRxiv paper is direct evidence this is not a theoretical concern.

**12.3.3 Clinical utility measurement (J).**
- **Decision Curve Analysis (DCA) / Net Benefit** — standard since Vickers & Elkin 2006, pipeline-standard by 2025.
- **Lancet Digital Health 2025** (S2589-7500(25)00098-6) — *Evaluation of performance measures in predictive AI models to support medical decisions: overview and guidance*. Authoritative 2025 summary.
- **AXIS-2 risk score** (BMC Emergency Medicine 2026, s12873-026-01496-w) — applied DCA to 30-day ED revisit prediction. Direct task-family overlap with this repo.
- **ACS-SRC readmission prevention** — DCA shows net benefit at 5–20 % risk range; "treat-all" and "treat-none" strategies as baselines.
- **Implication**: §11.1(e) MVIP should publish a DCA net-benefit array at fixed thresholds (e.g., 10 %, 15 %, 20 %, 30 %) as a required field of the `RiskAssessment` extension — not just AUROC / AUPRC metadata.

**12.3.4 Prospective CDSS trial design (K).**
- Pragmatic cluster-randomized: AIM ECG-AI (40276542); 38-clinic cognitive-impairment trial (PMC12551766).
- Pragmatic stepped-wedge: TBI VTE prophylaxis with external control sites (s13012-024-01386-4); Intermountain ePNa pneumonia stepped-wedge (AJRCCM).
- JMIR 2025 e63733 systematic review — *Designing CDSS: user-centered lens of design characteristics, challenges, implications*.
- Implication: the §11.1(e) reference implementation pairs naturally with a **pre-registered stepped-wedge simulated-evaluation plan** on Synthea — no live-patient arm required; the plan becomes the roadmap for a future prospective study.

**12.3.5 Drift detection in production (L).**
- Nat Commun 2024 / PMC10904813 — empirical data-drift experiments on real-world medical imaging.
- Systematic review *Strategies for detecting and mitigating dataset shift in ML for health predictions* (S1532046425001315) — model-based monitoring + statistical tests dominant detection; retraining + feature engineering dominant correction.
- **Key**: *"Monitoring performance alone is not a good proxy for detecting data drift."* Input-distribution drift needs its own detector.
- Implication: §11.1(e) MVIP must include a **drift-flag extension** (boolean + summary statistic) in the `RiskAssessment`. Even a simple KS / Wasserstein statistic on the 76-dim observation vector vs. training distribution is concrete and acceptable as a scaffold.

### 12.4 Verification status

| Claim | Source | Verification |
|:------|:-------|:-------------|
| Penda Health 16 % diagnostic / 13 % treatment error reduction, 39,849 visits, 15 clinics | arXiv:2507.16947 + OpenAI blog | ✓ WebFetch-verified (abstract); workflow-invocation + PHI-handling details deferred to full paper |
| kind-lab/mimic-fhir: FHIR R4, MIMIC-IV + ED, SQL+HAPI, no ML extensions | GitHub v2.1.0 (2024-10-28) | ✓ WebFetch-verified |
| Class-imbalance correction → miscalibration | medRxiv 2026.03.04.26347634 | Literature-cited (not fetched) |
| FDA PCCP 2025: 30 devices (≈ 10.2 %), 6 scenarios, Aug 2025 final | FDA media + innolitics year-in-review | Literature-cited |
| Colorado SB24-205 implementation 2026-06-30, healthcare high-risk | Seyfarth + Credo AI + GovTech | Literature-cited |
| EU AI Act medical high-risk effective 2027-08-02 | White & Case + Reed Smith + PMC11379845 | Literature-cited |
| STOPP/START v3 CDSS uptake 15 % and 62 % in two trials | PMC10447584 (v3 paper) + PSNet | Literature-cited |
| RxNorm integrates DrugBank, SNOMED CT, ATC, NDC, MeSH | NLM RxNorm overview + DrugBank blog | Literature-cited |
| MC Dropout ECE 0.76 → Deep Ensemble 0.8559 AUROC | arXiv:2511.18839 | Literature-cited |

### 12.5 Revised follow-up actions (supersedes §11.5)

1. **Do NOT rebuild the MIMIC→FHIR converter.** Extend **kind-lab/mimic-fhir** with an *ML extension profile module*: SHAP attribution, model version, training-data window, calibration strategy + epoch, uncertainty interval (Conformal or Deep Ensemble), fairness flag, drift flag, DCA net-benefit array. Ship as an add-on `ImplementationGuide`, not a fork. Address the MIMIC-III vs MIMIC-IV mismatch by providing a small MIMIC-III → MIMIC-IV schema adapter for this repo's LSTM-CNN input.
2. **Name the calibration strategy in the MVIP.** Default to *Conformal Prediction* for distribution-free guarantees; fallback to *Deep Ensembles* for parametric intervals. Explicitly reject Platt-scaling-only as inadequate under this repo's balanced-sampling training regime, citing medRxiv 2026.03.04.26347634 + arXiv:2410.18144 + Scientific Reports s41598-025-32656-w.
3. **Add a DCA net-benefit array to the `RiskAssessment` profile.** Fixed thresholds (10 %, 15 %, 20 %, 30 %); computed on the evaluation split. Include alongside AUROC / AUPRC in the model-version metadata extension. Reference Lancet Digital Health 2025 S2589-7500(25)00098-6 as the normative source.
4. **Extend the §10.4 GMLP audit checklist to a multi-jurisdictional matrix.** From "10 GMLP principles" to "GMLP + FDA PCCP presence + Colorado SB24-205 deployer controls + EU AI Act high-risk applicability + calibration strategy documented + drift monitor present." Per-principle tri-state scoring remains. Multi-jurisdictional score makes the audit paper stronger.
5. **Anticoagulation (warfarin × MIMIC-III) as the §11.2(e) first instantiation.** Baseline literature exists (PMC11784795 / PeerJ cs-2612) to compare against. Specific composite signal: *warfarin dose × readmission risk × CYP2C9/VKORC1 PGx × bleeding-risk ML*. Deferred until `MIMIC_DATA_ROOT` is set; the research design can be pre-registered now.
6. **Typed BC5CDR CHEMICAL → RxNorm normalisation layer.** Small standalone artifact. Unblocks interoperability with FHIR `MedicationRequest` and with the RxNorm DDI API (DrugBank-powered). Data-free; uses NLM RxNav.
7. **Drift-flag scaffold in the reference implementation.** Simple KS / Wasserstein on the 76-dim observation vector vs. the training distribution. Publishing a scaffold, not a SOTA detector. The *presence* of a drift field in the extension IG is the contribution.
8. **Adherence-prediction multi-head extension (medium-term).** Add an adherence prediction head alongside the readmission head to target the 35 % non-adherence sub-category of drug-related readmissions (§11.2(a)). Literature baselines: NYU HF cohort JAMIA ocaf162, MedAdhereAI medRxiv 2025.07.01. Data-required.
9. **Pre-register a stepped-wedge simulated-evaluation plan on Synthea.** Pairs with §11.1(e) primary deliverable; references AJRCCM ePNa + Implementation Science s13012-024-01386-4 as design precedents.
10. **Reject scope creep** into drug repurposing, imaging CDS, genomics variant calling — these exist in the literature but are orthogonal to this repo's readmission-prediction foundation and the two axes §11 named.

### 12.6 Scope & deferral notes

- **Explicit defers** (noted, not executed): (a) MIMIC-III-specific MIMIC→FHIR adapter audit — all four major OSS projects target MIMIC-IV; this repo is MIMIC-III, so either a schema adapter or a narrow custom path is required; captured in §12.5 action 1. (b) International regulatory beyond US + EU (KR MFDS, JP PMDA, UK MHRA) — only lightly searched in the preceding rounds; expand if a Korean / Japanese / UK deployment scenario becomes real.
- **Out of scope**: drug repurposing, imaging CDS, genomics variant calling, billing / revenue-cycle CDS, supply-chain pharmacy robotics. These are real fields with active literature but are orthogonal to the readmission + medication safety focus established in §10 + §11.
- **Source reliability**: several 2025 sources are pre-prints (medRxiv / arXiv). Where a load-bearing claim rests on a pre-print (notably §12.3.2 miscalibration, §12.3.5 drift review), final-publication peer-reviewed versions must replace the pre-print citation before any paper built on §11.1(e) is submitted.

---

## 13. AI system design & management — MIT SDM lens (2026-04-21)

### 13.0 Pivot — why this section exists

§10–§12 framed the problem from the *clinical-domain* side (FHIR resources, CDSS standards, medication-safety patterns, regulatory checklists) and asked "how does this model deploy into an EHR-compatible hospital environment?" §13 re-frames the same problem from the *engineering-systems + management* side and asks the questions an MIT SDM (System Design & Management) / ESD (Engineering Systems Division) analysis demands. The clinical facts from §10–§12 become inputs; the artifacts of this section are the deliverables:

- What *is* the system and where are its boundaries?
- Who are the stakeholders and what value flows among them?
- Which quality attributes drive architecture, and how do they trade off?
- Which feedback loops govern system behaviour?
- Which decisions are *commitments* and which are *real options*?
- Where is the field in its technology lifecycle, and what does dominant-design analysis predict?
- Who owns what, and what operating model supports it?
- What is the portfolio view under dependency + risk?
- What are the TCO, unit economics, and value model?
- How does a sociotechnical-transition (MLP) analysis position the system?

### 13.1 System boundary & OPM view

Three boundary choices, each implying different stakeholders and quality-attribute concerns:

| Boundary | Contents | Excluded | Used by |
|:---------|:---------|:---------|:--------|
| **Model boundary** | LSTM-CNN + typed BC5CDR NER + SHAP | Everything else | Single-paper academic evaluation (what the original repo optimises) |
| **Deployment boundary** | Model + FHIR adapter + CDS Hooks service + drift monitor + calibration layer + SMART auth | EHR, clinician, patient, billing | §11.1(e) MVIP scope |
| **Clinical-work-system boundary** | Deployment + EHR + clinician + patient + workflow + audit policies + regulators | Population health, payer analytics | SDM analysis in §13 |

OPM (Object-Process Methodology, Crawley / Dori) sketch:
- **System function**: *reduce preventable 30-day readmissions by informing discharge decisions and downstream medication-safety steps.*
- **System form**: 4 modalities → predictor → CDS Hooks card → clinician decision → medication order → medication administration → outcome.
- **Decomposition**: the model is a *constituent*; the EHR is the *environment*; the clinician is both *operand* (target of information) and *operator* (agent of decision); the patient is the *beneficiary*.

Load-bearing insight: the model is *not* the system. The model is a constituent that enables a system function at the clinical-work-system boundary. Most ML papers stop at the model boundary; SDM analysis forces the outer ring.

### 13.2 Stakeholder Value Network (SVN)

Primary stakeholders (direct value exchange):
- **Patient** — receives outcome (value-in); contributes data (value-out).
- **Clinician / prescriber** — receives risk information + alerts (value-in); contributes attention + decisions + override signal (value-out).
- **Hospital administration** — receives reduced readmissions + regulatory compliance (value-in); contributes capital + governance + operations (value-out).
- **EHR vendor** (Epic / Oracle Health) — receives app-ecosystem value (value-in); contributes integration platform (value-out).
- **Regulator** (FDA + state CO + EU + FDA/EMA drug-AI) — receives compliance artifacts (value-in); contributes legitimacy-to-deploy (value-out).
- **Payer** — receives reduced readmission cost (value-in); contributes reimbursement policy (value-out).

Secondary: model developer / researcher, data steward (MIMIC / PhysioNet), IRB / ethics, IT-ops / SRE, pharmacy (downstream medication-safety consumer).

**Load-bearing value flow**: *clinician attention* is the scarce good in the system. Alert fatigue (§11.2) erodes it; trust is the currency that gates whether the clinician spends attention on the model's output. Every architectural choice (calibration, explainability, alert prioritization) either protects or depletes the clinician-attention budget. Literature converges here — *stakeholders favour models endorsed by trusted people and institutions* (JMIR 2025 e78757) — so SVN points at trust as the network's critical edge.

### 13.3 Quality Attribute Scenarios (QAS) + ATAM-style tradeoffs

ATAM (Kazman / Klein / Clements, SEI) adapted to clinical ML. For each quality attribute, a concrete scenario (source → stimulus → environment → response → measure):

| Quality attribute | Stimulus | Source | Environment | Response | Measure | Anchor |
|:------------------|:---------|:-------|:------------|:---------|:--------|:-------|
| Reliability | CDS Hooks request | Clinician at order-entry | Production EHR | Prediction returned | < 500 ms p99 | CDS Hooks spec |
| Availability | Service failure | Infra outage | Business hours | Graceful fallback | 99.95 % uptime | GMLP P10 |
| Safety | Silent input-drift > ε for > N days | Population shift | Production | Drift flag + operator alert | KS statistic > threshold → alert < 24 h | §12.3.5 |
| Modifiability | FHIR R4 → R5 migration | Vendor roadmap | Upgrade window | No code rewrite | < 1 PR + config | §11.1(a) |
| Security | Untrusted SMART launch | Hostile app | Prod | Scope-limited access | 0 cross-patient reads | SMART 2.1 |
| Fairness | Subgroup audit | Audit trigger | Any | Per-subgroup AUROC + spread | Spread < 0.05 | GMLP P3 |
| Explainability | Clinician inspection | User action | Chart view | SHAP top-5 features + force plot | < 200 ms to render | README "Key finding" |
| Maintainability | Drift flag raised | Monitor | Prod | Retrain + PCCP-aligned redeploy | < 30 days | §12.5 action 7 |

Tradeoff identification (ATAM step: sensitivity + tradeoff points):
- **Reliability ↔ Explainability**: SHAP DeepExplainer is O(model-passes × features); the 500 ms p99 budget tightens against the 200 ms render budget. *Mitigation*: pre-compute and cache SHAP at prediction time; serve both in one response.
- **Modifiability ↔ Safety**: Conformal-Prediction wrapping simplifies model-swap but widens prediction intervals. *Mitigation*: calibrate the Conformal set on a production-aligned reference sample.
- **Fairness ↔ Overall accuracy**: subgroup-spread constraint adds a Lagrangian term during training. *Tradeoff accepted*: small AUROC give-up for fairness guarantee (GMLP P3).
- **Availability ↔ Change-velocity**: PCCP raises update velocity but requires more pre-registration. *Tradeoff accepted*: structured envelope of changes.

### 13.4 System dynamics — Causal Loop Diagram (CLD)

Four loops govern system behaviour. R = reinforcing, B = balancing.

**R1 — Alert fatigue (undesired reinforcing).** More alerts fired → override rate rises → the alert channel signal-to-noise drops → model developer tunes toward higher sensitivity to avoid missing → more alerts fired → (reinforces). **Leverage point**: prioritization conditioned on *patient-specific* outcome risk (§11.2(e)), raising the expected value per alert click.

**B2 — Trust erosion (undesired balancing).** Bad prediction → trust drops → adoption drops → fewer overrides observed → slower feedback → slower recalibration → more bad predictions. **Leverage point**: explainability (SHAP) compresses the feedback cycle by letting the clinician infer *why* a prediction was wrong without waiting for aggregate statistics. This is the empirical finding from this repo's README ("Key finding — Interpretability enabled clinical adoption").

**R3 — Regulatory ratchet (field-level reinforcing).** Capability grows → regulation tightens → compliance cost rises → fewer deployments survive → slower learning → pressure builds for fresh capability → (reinforces). **Leverage point**: PCCP-style real options (§13.7) absorb regulatory cost into pre-registered envelopes, damping the ratchet.

**R4 — Platform network effect (desired reinforcing).** More models expose SMART + CDS Hooks + HMCP → more value to EHRs from supporting the protocol → EHR vendor support and docs improve → easier for next model to expose → more models expose → (reinforces). **Leverage point**: this repo's §11.1(e) MVIP *participates* in this loop; adoption is the contribution.

Cross-loop note: R1 and B2 are tightly coupled by the clinician-attention budget. The SDM-correct architectural move is to *design against R1 by conditioning on outcome risk while designing against B2 by preserving explainability* — i.e., both §11.1(e) and §11.2(e) attack the same sociotechnical bottleneck from different sides.

### 13.5 Architecture Decision Records (ADRs)

Eight ADRs capture the major §11–§12 architectural decisions. Each should become a markdown file under `docs/adr/` (§13.15 action 1).

| ADR | Decision | Status | Consequences (+/−) |
|:----|:---------|:-------|:-------------------|
| **001** Reuse `kind-lab/mimic-fhir` vs. build custom | Reuse + extend with ML IG (§12.5.1) | Accepted | + inherits 25 FHIR profiles; − locked to MIMIC-IV schema, requires MIMIC-III↔IV adapter |
| **002** Calibration strategy for balanced-sampling model | Conformal Prediction default; Deep Ensembles fallback (§12.5.2) | Accepted | + distribution-free guarantees; − wider intervals than parametric; explicitly rejects Platt-only |
| **003** Interface triple (CDS Hooks + SMART + HMCP) vs. single interface | Triple (§11.1(e)) | Accepted | + three consumer modalities (workflow-embedded / launched-app / LLM-orchestrated); − 3× surface to maintain |
| **004** MIMIC-III primary vs. migrate to MIMIC-IV | Undecided — decision node | Open | MIMIC-IV has active FHIR tooling (§12.1.3); MIMIC-III has direct anticoagulation baseline (§12.2.1). Defer pending data access |
| **005** FHIR server for reference impl | HAPI FHIR (§12.1.6) | Accepted | + research-friendly, 2-command Docker; − Java dev cost; Aidbox is the production-pilot upgrade path |
| **006** Retain balanced sampling | Retain + document with calibration strategy (ADR-002) | Accepted | + preserves existing training pipeline; − imposes non-trivial deployment calibration work that ADR-002 discharges |
| **007** Drug-entity normalisation | BC5CDR CHEMICAL → RxNorm (§12.5.6) | Accepted | + EHR MedicationRequest interop, DrugBank-via-RxNorm DDI API; − NLM RxNav runtime dependency |
| **008** Scope — platform vs. product | Platform (reusable spec + IG + reference impl) demonstrated on this repo's LSTM-CNN (§13.14) | Accepted | + other research groups can adopt; − requires abstraction work not in the original repo plan |

ADR-004 is deliberately left open as a decision node — SDM practice holds that unresolved decisions *should be visible*, not hidden.

### 13.6 Design Structure Matrix (DSM) — capability dependency

Component dependencies (row *depends on* column). "↓" = build-time dep, "→" = runtime data flow.

|   | LSTM | NER | BC5CDR | SHAP | FHIR-adapter | Calib | Drift | RxNorm | CDS-Hooks | MCP |
|---|:----:|:---:|:------:|:----:|:------------:|:-----:|:-----:|:------:|:---------:|:---:|
| **LSTM** | — | — | — | — | → | ↓ | ↓ | — | → | → |
| **NER** | — | — | ↓ | — | → | — | — | ↓ | → | → |
| **BC5CDR** | — | — | — | — | — | — | — | — | — | — |
| **SHAP** | ↓ | — | — | — | — | — | — | — | → | → |
| **FHIR-adapter** | ↓ | ↓ | — | ↓ | — | — | — | ↓ | ← | ← |
| **Calibration** | ↓ | — | — | — | — | — | ↓ | — | — | — |
| **Drift** | ↓ | ↓ | — | — | → | ↓ | — | — | — | — |

Readings:
- **FHIR-adapter has fan-in 4** (LSTM, NER, SHAP, RxNorm) — it is the integration bottleneck; failure here cascades. *Implication*: the adapter is the critical path for ADR-003's triple interface.
- **Calibration and Drift form a short chain** (both depend on LSTM; Drift further depends on Calibration). *Implication*: deploy coherently; do not ship the model without both.
- **SHAP has fan-out 2** (CDS Hooks, MCP) — compute once, cache, reuse. *Implication*: explainability cache is an optimization point against the Reliability ↔ Explainability tradeoff in §13.3.
- **BC5CDR has zero out-arrows** in the deployment-time DSM — it is a build-time artifact (the model used to produce the typed NER), not a runtime dependency. *Implication*: the deployment stack need not ship `en_ner_bc5cdr_md` at runtime; embeddings or cached outputs suffice.

### 13.7 Real options in the design

Four options, each adds flexibility at a capped scaffolding cost:

| Option | Exercise when | Exercise cost | Value when exercised | Scaffolding cost (now) |
|:-------|:--------------|:--------------|:---------------------|:-----------------------|
| Extend to FHIR R5 subscriptions | EHR vendors adopt R5 subscriptions | Config + test; via R5 Subscription Backport IG | Real-time event-driven invocation replaces polling | Include Backport endpoint stub in MVIP |
| Activate adherence multi-head | `MIMIC_DATA_ROOT` becomes available | Add head + retrain | Targets 35 % non-adherence slice of drug-related readmissions (§11.2(a)) | Schema slot in MVIP; no training |
| PCCP post-clearance | Model performance drifts or clinical need shifts | Execute pre-registered modification protocol | Iterative improvement without full re-submission. **Only ~8 % of 2025 cleared AI devices exercised this** (medRxiv 2025.08.26.25334477) — the option is under-exercised, so early adopters gain relative advantage | Write PCCP draft alongside model card |
| Swap base model via Conformal wrapper | New backbone (transformer) becomes available | Re-fit Conformal calibration set | Upgrade without API break | Design LSTM-CNN as wrapped; API-stable |

Real-options framing reframes "over-engineering" critiques: each option *does not require* immediate implementation. The investment is the scaffolded slot (HMCP manifest, Subscription endpoint, multi-head architecture, Conformal wrapper) — not the exercise.

### 13.8 Technology lifecycle & dominant design

Utterback S-curve analysis of clinical-AI-in-hospital-EHR integration:

- **Dominant-design candidate**: SMART on FHIR + CDS Hooks + FHIR R4/R5 + MCP/HMCP. §12.1.1–.2 evidence shows Epic + Oracle Health + emerging open-source MCP stack (WSO2 / AgentCare / Momentum / Innovaccer HMCP) all converging on this quadruple.
- **S-curve position**: early-adopter → early-majority transition. Divergent indicators:
  - Infrastructure high: 90–95 % of U.S. hospitals have SMART endpoints; 100 % of ONC-certified EHRs do (§11.1).
  - Governance low: only ~12 % of hospitals have a formal AI governance framework (NIST AI RMF Censinet 2025); only ~8 % of cleared AI devices use PCCP.
  - This "infrastructure ahead of governance" gap is characteristic of the steep middle of the S-curve.
- **Disruption watch**: LLM orchestrators (GPT-style) positioning between clinician and EHR as "system of work" (McKinsey 2025); hyperscalers (Azure / AWS / Google) consolidating the operating layer.
- **Implication**: deploying *into* the emerging dominant design is strategically correct. §11.1(e) is this action. A strategy that waits for the dominant design to stabilize (e.g., targets FHIR R6 only, or waits for a standardized HMCP) forgoes the early-majority window.

### 13.9 Organizational / responsibility design (management lens)

RACI for a deployed ML-CDSS. A = Accountable, R = Responsible, C = Consulted, I = Informed.

| Activity | CMIO / CHIO | CIO | Clinical dept | IT-Ops / SRE | Data science | Ethics / compliance | Vendor |
|:---------|:----:|:---:|:-------------:|:------------:|:------------:|:-------------------:|:------:|
| Clinical appropriateness of output | **A** | C | R | I | C | C | I |
| Model training / retraining | C | I | C | I | **R/A** | C | C |
| FHIR / CDS Hooks integration | C | **A** | I | R | C | I | R |
| Drift monitoring | C | C | I | R | **R/A** | I | C |
| Override-rate monitoring | **R** | I | R | C | C | **A** | I |
| Fairness audit | C | I | C | I | R | **R/A** | I |
| PCCP submission | C | C | C | I | R | **R/A** | C |
| Incident response | **A** | R | R | R | R | C | C |

Key reading: **override-rate monitoring is cross-functional and often under-owned**. CMIO / CHIO must be named Accountable because override rate is the clinical-safety-critical feedback signal for the entire system (ties back to CLD loop R1 in §13.4).

Team topology (Skelton & Pais). The clinical ML capability maps to a **platform team** serving **stream-aligned teams** (clinical departments with specific use cases — readmission, sepsis, AMS, etc.) plus an **enabling team** (ethics + data governance + compliance). The "shadow AI" trend flagged by Wolters Kluwer 2026 is best read as a *missing platform service* symptom, not a policy-compliance failure — the platform team's mandate is to provide sanctioned alternatives.

CHIO (Chief Health Informatics Officer) is the emerging C-suite placement for this function (TechTarget 2025), consolidating CMIO + CNIO accountabilities.

### 13.10 Portfolio view & roadmap

Capability portfolio with §12.5 + §13.15 actions (rows = capability, cols = quarter):

| Capability | Q2 '26 | Q3 '26 | Q4 '26 | Q1 '27 | Q2 '27 |
|:-----------|:------|:------|:------|:------|:------|
| Readmission ref impl (§11.1(e)) | **Primary** scaffold (ADRs + MVIP spec) | Reference impl stubs | Synthea demo | Stepped-wedge plan | Early pilot |
| GMLP + multi-jurisdictional audit | Extend pool | Dry-run | Scoring rubric | Draft paper | Submit |
| Outcome-risk-conditioned med safety (§11.2(e)) | — | Data-access negotiation | Warfarin sim design | Warfarin sim | DDI / recon / PGx |
| NER → RxNorm layer | — | Scaffold | Validate | Release | Reuse downstream |
| Drift scaffold (§12.5.7) | — | Scaffold (KS) | Integrate | — | — |
| Adherence multi-head | — | — | — | Feasibility | Build (data-req.) |
| Stepped-wedge plan on Synthea | — | — | Draft | Pre-register | — |
| Model card + datasheet (§13.15.3) | Draft | Publish | Review annually | — | — |

**Critical path**: §13.15 action 1 (ADRs) → action 2 (MVIP spec) → action 3 (model card + datasheet) → action 7 (CDS Hooks service stub) → Synthea demo → stepped-wedge plan → early pilot. This path mints a deployable artifact and a reviewable paper simultaneously.

### 13.11 NIST AI RMF mapping (Govern / Map / Measure / Manage)

| RMF function | This-repo-specific practices |
|:-------------|:-----------------------------|
| **Govern** | ADR-008 sets platform-vs-product scope; §13.9 RACI names Accountable roles; external alignment: NAM *AI Code of Conduct* (May 2025), CAOS-RE (Springer 10.1007/s10728-025-00537-y); publish ADRs + MVIP spec + model card as governance artifacts. |
| **Map** | §13.1 system boundary; §13.2 SVN; §13.3 QAS (stakeholders × environments × responses); §13.4 CLD identifies feedback-loop risks; §13.13 MLP positions the niche. |
| **Measure** | §13.3 QAS response-measures + §12.5.3 DCA net-benefit array + subgroup AUROC + override rate + drift KS statistic + calibration ECE. Publish measured values in the `RiskAssessment` model-version extension. |
| **Manage** | Drift → retrain workflow (§12.5.7); PCCP pre-registered envelope (§13.7); RACI-bound incident response (§13.9); real-option exercise triggers (§13.7) handled at Manage-phase review. |

Only ~12 % of hospitals currently have a formal AI governance framework (Censinet 2025, single-source — verify before citing in paper). Together §13.9 RACI + §13.3 QAS + §13.4 CLD + §13.11 RMF map constitute a **deployable governance scaffold** for a hospital adopting this repo's model — i.e., the hospital gets the governance artifacts alongside the code.

### 13.12 Economics — TCO + unit economics + NPV

TCO sketch for a single-hospital deployment of this repo's LSTM-CNN (nominal ranges):

| Category | Annual $ (order of magnitude) | Notes |
|:---------|:------------------------------|:------|
| Infrastructure (CPU / GPU + FHIR server hosting) | $10 K – $30 K | HAPI + light serving; cloud surcharge if hyperscaler |
| Model serving + inference compute | $5 K – $20 K | Batch or near-real-time; < 50 predictions / min |
| Drift monitoring + telemetry | $3 K – $10 K | §12.5.7 |
| Periodic retraining + PCCP ops | $20 K – $60 K | Data-science FTE fraction |
| Governance (ethics / compliance / legal) | $10 K – $30 K | Shared overhead |
| Integration (EHR adapter + on-call) | $30 K – $100 K | Upfront-heavy; amortize |
| **Total nominal annual OPEX** | **$78 K – $250 K** | |

**Vendor hidden costs** (§13 search): a healthcare deployment found 63 % of real AI costs came from data-pipeline optimisation + GPU-cluster management — invisible in vendor ROI calculators. Any SDM TCO model must internalize these.

Value model:
- preventable readmissions × cost-per-readmission × PPV × adoption-rate
- Typical U.S. readmission cost: $15 K – $20 K per event
- On a base of 15 % readmission rate × 5 000 discharges / year, a 1 pp reduction = 50 events × $17 K = **$850 K / yr** — comfortably exceeding the $78 K – $250 K TCO band if PPV × adoption > ~ 25 %.
- Upper-bound precedent: RPM (remote patient monitoring) has shown 20–40 % readmission reduction (industry reports, 2025).
- Industry ROI benchmark: $3.20 per $1 over 14 months; 147 % 3-year ROI (healthcare-AI vendor reports; treat cautiously — §13.16).

**Real-options NPV adjustment**. The PCCP real option (§13.7) is Black-Scholes-style: a pre-registered modification protocol is a *call option on future model improvements*, valuable precisely when model-performance uncertainty is high. Under 2025 FDA guidance, exercising it costs only the protocol validation step — orders of magnitude cheaper than a new 510(k). The option premium is what the repo pays now (ADR + documentation work); the pay-off is realized in the 2027–2029 window.

### 13.13 Sociotechnical transition — MLP (Geels)

- **Landscape** (stable macro-pressures): value-based care payment reform; aging population; clinician shortage; regulatory tightening; AI capability curve; geopolitical data-sovereignty pressure.
- **Regime** (current sociotechnical system, self-reinforcing): fee-for-service billing; EHR-vendor-dominated IT; rule-based CDS; physician-central workflow; episodic care delivery.
- **Niches** (innovation loci): academic research repos (this repo's origin); startup LLM copilots (Penda Health + OpenAI §12.1.1; Abridge; Nuance DAX); EHR-vendor AI modules (Epic's ~200 AI features); regulatory sandboxes (UK MHRA, Korea K-FDA).
- **Transition pressure**: strong landscape pressure on the regime (VBC + demographics + shortage) + accelerating niche maturation (LLM + agentic AI) → *regime-destabilizing*. Transition window estimated 2026–2030.

Position of §11.1(e): explicitly a **niche-to-regime bridge**. It gives the niche (a research-grade LSTM-CNN + typed NER) a standardized interface (SMART + CDS Hooks + MCP) into the regime's dominant design. The strategy is not to replace the regime — it is to plug into the regime's dominant-design integration point so niche innovations scale at regime cadence. The SDM-correct move is to *ride* the transition, not fight it.

### 13.14 Reframed primary thread — from paper to platform

The SDM lens converts the §11.1(e) deliverable:

- Before (§11.1(e) + §12.5 language): "reference integration architecture" = spec + paper + minimal reference impl.
- After (§13 language): a **platform product** delivered as three layered artifacts:
  1. **Specification layer** — ADRs, MVIP spec, FHIR IG extensions (DCA / Conformal / drift / model-version / fairness).
  2. **Reference implementation layer** — HAPI stack, CDS Hooks service, SMART launch, HMCP manifest, Conformal wrapper, drift detector, RxNorm normalizer.
  3. **Documentation layer** — model card (Mitchell 2018), datasheet (Gebru 2018), ADRs, RACI, QAS, TCO model, roadmap, CLD.

Each layer has a matching SDM artifact: spec governed by ADRs + QAS; reference impl validated against QAS response-measures; documentation instantiates NIST AI RMF's Map + Govern functions. The platform framing makes *other research groups* the primary adopter — success is measured when a second MIMIC-derived legacy model wraps itself using the spec.

### 13.15 SDM-style action list (supersedes §12.5)

**Specification + governance (data-free)**:
1. Write **ADR-001 … ADR-008** as markdown under `docs/adr/` (§13.5 content).
2. Write the **MVIP (Minimum Viable Integration Profile) spec** covering (a) SMART launch context, (b) CDS Hooks service manifest + Card schema, (c) HMCP tool manifest, (d) `RiskAssessment` FHIR extensions (SHAP attribution, model version, training-data window, calibration strategy, uncertainty interval, fairness flag, drift flag, DCA net-benefit array).
3. Write a **model card** for the LSTM-CNN and a **datasheet** for the MIMIC-III-derived features.
4. Write the **QAS set** (§13.3) as `docs/qas.md` with explicit response-measures.
5. Draw the **CLD** (§13.4) and publish as `docs/system-dynamics.md` with the explicit mapping of §12.5 + §13.15 actions → CLD leverage points.
6. Instantiate **NIST AI RMF** mapping (§13.11) as `docs/ai-rmf.md`.

**Reference implementation (code scaffold, data-free)**:
7. FastAPI **CDS Hooks service stub** returning a dummy `RiskAssessment` with the full MVIP extension set.
8. **Conformal Prediction wrapper** around the existing LSTM-CNN (ADR-002).
9. **KS / Wasserstein drift detector stub** on the 76-dim observation vector (§12.5.7).
10. **BC5CDR CHEMICAL → RxNorm normalizer** via NLM RxNav (ADR-007, §12.5.6).
11. **SMART on FHIR launch bootstrap** + sample Epic App Launcher manifest.

**Validation & pilot (semi-data-free via Synthea)**:
12. **Synthea → FHIR → model-input adapter** bridging kind-lab MIMIC-IV profiles to this repo's MIMIC-III input contract (documents ADR-004 decision).
13. **End-to-end demo**: Synthea sample patient → SMART launch → CDS Hooks invocation → model → Card → SHAP render.
14. **Pre-register a stepped-wedge simulated-evaluation plan** on Synthea, patterned on ePNa AJRCCM + Implementation Science s13012-024-01386-4.

**Parallel audit track**:
15. Extend §10.4 GMLP audit to a **multi-jurisdictional score** (FDA GMLP + PCCP presence + CO SB24-205 + EU AI Act + calibration-strategy documented + drift-monitor present + model-card present) per §12.5.4.

**Deferred (data-required)**:
16. Warfarin × readmission simulation per §12.5.5 — §11.2(e) first instantiation.
17. Adherence multi-head per §12.5.8.

**Rejected / out of scope**:
18. Drug repurposing, imaging CDS, genomics variant calling, population-health dashboards, billing/RCM CDS.

### 13.16 Verification & source reliability notes

| Claim | Source | Reliability |
|:------|:-------|:------------|
| Only 12 % of hospitals have formal AI governance framework | Censinet 2025 blog | Single-source; verify via CHAI / HIMSS survey before paper submission |
| ~8 % of 2025 cleared AI devices use PCCP | medRxiv 2025.08.26.25334477 | Pre-print; wait for peer-reviewed |
| 82 % / 74 % / 79 % predictive-AI hospital evaluation rates | AHA Market Scan 2025-11-04 | Industry survey; treat as directional |
| $3.20 per $1 / 14-month ROI | Multiple industry reports | Vendor-marketing-adjacent; for academic use cite case studies (RPM 20–40 % readmission reduction) instead |
| NAM *AI Code of Conduct* May 2025 | NAM primary source | Reliable |
| CAOS-RE framework | Springer *Health Care Analysis* 10.1007/s10728-025-00537-y | Peer-reviewed |
| McKinsey "modular architecture" | McKinsey 2025 industry report | Reliable framing; not peer-reviewed |
| Epic U.S. acute-care market share 37.7–39.1 % | Multi-source (healthcarereimagined, SPsoft) | Triangulated, reliable |

### 13.17 MIT SDM framing — artifact ↔ coursework map

The artifacts above map onto SDM / ESD coursework:

| Artifact in §13 | MIT SDM / ESD anchor |
|:----------------|:---------------------|
| OPM + system boundary (§13.1) | ESD.34 System Architecture (Crawley) |
| SVN (§13.2) | Stakeholder Value Network analysis (Cameron / Rebentisch) |
| QAS + ATAM (§13.3) | ATAM (Kazman, SEI) adopted in SDM system-architecture sequence |
| CLD (§13.4) | 15.871 System Dynamics (Sterman), ESD.68 Modeling & Assessment |
| ADRs (§13.5) | Architecture Decision Records (Nygard); paired with ESD.34 decision analysis |
| DSM (§13.6) | ESD.36 Product Development (Eppinger) |
| Real options (§13.7) | 16.888 Real Options in Engineering Systems (de Neufville) |
| Dominant design / S-curve (§13.8) | Technology & Innovation (Utterback, Cusumano) |
| RACI + team topology (§13.9) | Organizational design (Malone); Team Topologies (Skelton) |
| Portfolio + roadmap (§13.10) | Technology Roadmapping (Phaal); engineering-management electives |
| NIST AI RMF (§13.11) | Risk Management (MIT Sloan ESD.71) |
| TCO + NPV (§13.12) | 15.401 Finance Theory; engineering-economics electives |
| MLP (§13.13) | ESD.864 Technology Strategy (Geels / Schot sociotechnical transitions) |
| Platform-vs-product reframing (§13.14) | Platform Strategy (Cusumano / Gawer) |

The deliverable reads as an SDM thesis capstone artifact: an engineered system characterised by multi-framework analysis with a concrete management recommendation (§13.14) and a sequenced action plan (§13.15).
