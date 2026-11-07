# Initial BreaKHis pruning research protocol

Version 1, 7 November 2026. **Draft; no scientific fits or confirmatory freeze.**
The machine-readable source is [protocol.initial.json](../../configs/research/protocol.initial.json).
Its [decision register](../../configs/research/decisions.initial.json) preserves
revisions; the [literature matrix](../../configs/research/literature.initial.json)
records access and unresolved overlaps. [M0.5 API](../specs/m0/M0_5_PROTOCOL_API.md)
defines validation and binding. The [research guide](Breast_Histopathology_Pruning_End_to_End_Research_Guide.md)
and [roadmap](RESEARCH_IMPLEMENTATION_ROADMAP.md) explain scientific intent.

## Scope and interpretation

The working scope follows the existing BreaKHis-only, binary sprint plan and the
user's CPU/RTX 5070 Ti deployment targets. All 17 template topic records remain
**proposed**, because they include design choices requiring scientific review.
This does not revoke previously authorized engineering work or imply that those
capabilities have been implemented. Binding evidence never changes a proposal to
an agreement. `research_owner` is an unresolved role, not an invented reviewer.

| Topic | Initial policy and open choice | Review deadline |
| --- | --- | --- |
| Task/source | Benign/malignant, all available magnifications, retained v1 binary derivative; measure dimensions and review transformation lineage | M1 |
| Grouping | Full YEAR-SLIDE_ID with suffixes; duplicate-linked components before partitions; filename/slide language until clinical mapping exists | M1 |
| Aggregation | Proposed within-magnification mean then equal mean over available magnifications; secondary ordinary image mean; no imputation | M1 |
| Hardware | GPU training; CPU batch-one median forward latency primary; GPU timing secondary; two CPU threads provisional | M2 |
| Architecture | Custom eight-convolution VGG primary; proposed MobileNetV3-small dense comparator, ResNet18 subject to measured cost; VGG surgery first | M2 |
| Evaluation | Proposed five outer folds; group-separated inner training/selection/calibration roles; exact support compromise remains open | M1 |
| Seeds | Separate split/training/initialization/loader/augmentation/pruning/search/bootstrap/timing namespaces; numerical seeds null until recorded | M1 |
| Selection | Distinct development-only epoch/candidate selection and threshold calibration; no outer-test feedback | M1 |
| Safeguards | Sensitivity/specificity margins remain null; report absolute performance/support/uncertainty; no implied clinical noninferiority | M1 |
| Budgets | Dense/adaptation/recovery epochs, candidates and latency budgets null; match data, threshold opportunities and recovery/search accounting | M2 |
| Compute gate | Day-3 cost estimate; reduce scope to a disclosed pilot if the measured plan exceeds the 7-9 day window | M2 |
| Hypotheses | O1 concentrated damage; O2 incremental mechanism beyond controls; O3 architecture/device robustness within BreaKHis | M5 |
| Comparisons | A/B/C/D/E with matched constraints/costs; uniform/random/scratch/compact controls; honest negative-result route | M5 |
| Statistics | Group-paired uncertainty, class support, fold/seed separation and limited prespecified comparisons | M1 |
| Endpoints | Group sensitivity with specificity plus CPU latency; secondary accuracy/aggregation/size/MAC/GPU results; declared timing regions/sessions | M1 |
| Claims | No verified-patient, clinical-safety, established-novelty or independent-dataset claim; confirmatory freeze belongs to M6 | M5 |
| Pretraining | Recipe/weights unresolved; record normalization/version; fitted transforms use training data only | M2 |

Aggregation is a design recommendation, not a promise of greater accuracy. Each
group contributes one observation to group-level metrics. Report missing
magnification/class/subtype support and per-magnification denominators. Training
sampling or malignant loss weighting is a separate intervention.

## Decisive comparisons

| Method | Ranking/allocation and selection role |
| --- | --- |
| A | Conventional magnitude channel pruning with ordinary development selection |
| B | Same pruning as A with sensitivity/specificity-constrained selection |
| C | Malignant weighting with B's selection opportunities |
| D | Proposed group-balanced vulnerable-case mechanism with B's constraints |
| E | Strongest feasible established adaptive/class-aware comparator with B's constraints |

Equalize adaptation, recovery and search accounting and report costs explicitly.
No numerical tolerance is an agreed clinical margin. Keep epoch/candidate choice
separate from operating-threshold calibration and report the argmax/0.5 reference
alongside development-selected operating points. If D does not repeatably improve
on B/C/E, revise the mechanism or report an empirical/negative finding; additional
seeds do not create more independent groups or rescue a missing contribution.

## Data, exposure and evidence boundaries

Clinical identities and derivative lineage remain unresolved until attributable
evidence is reviewed. Historical real-image exposure propagates across linked
components and magnifications. New seeds cannot make an exposed group untouched.
Missing logs never establish no exposure. A lineage exception needs an explicit
scope decision; an engineering pass does not approve real fits.

At publication into an M0 run, bind the protocol to the source snapshot, audit
inventory bytes, exposure register bytes, decision input and literature input.
Stale supplied bindings fail. No operator input file or older run is overwritten.
The full evidence graph and Gate G0 remain M0.6. Final freeze before held-out
evaluation remains M6 and is forbidden as an M0 lifecycle.

## Literature status and limits

The initial matrix contains seven review targets, using primary pages checked on
7 November 2026. It is a focused starting matrix, not an exhaustive survey.

| Target | Actual access in this implementation |
| --- | --- |
| [NetAdapt](https://web.mit.edu/netadapt/) | Author project abstract/bibliography; full methods pending |
| [CATRO v2](https://arxiv.org/abs/2110.10921v2) | Author abstract/bibliography; journal DOI distinguished from preprint version |
| [Attend Who is Weak](https://pmc.ncbi.nlm.nih.gov/articles/PMC10089697/) | Focused reading of accessible methods/settings; independent overlap and reproduction adjudication pending |
| [MICCAI fairness pruning](https://papers.miccai.org/miccai-2024/040-Paper2133.html) | Official abstract/bibliography; linked full PDF not reviewed |
| [DATE class-aware pruning](https://github.com/HWAI-TUDa/Class-Aware-Pruning) | Author repository/README only; faithful reproduction not established |
| [TKDD goal-specific pruning](https://doi.org/10.1145/3721293) | Publisher inaccessible through browser tool; guide target/citation requires review |
| [ChannelZip](https://doi.org/10.1145/3729534) | Publisher inaccessible through browser tool; abbreviated target, exact citation unresolved |

Access level is not review completion. The matrix retains unknown fields and
reasons, owner/deadline, costs and selection questions, code/license uncertainty
and unresolved overlap. Accessible code does not prove faithful reproduction.
Central review gaps restrict novelty/comparator freeze; none establishes novelty.

## Updating decisions

Edit a working copy of the inputs. For a changed topic, retain the previous
decision row, append the next consecutive revision with `supersedes` set to
`decision_id@previous_revision`, and update the matching protocol topic. Keep the
stable ID/topic, rationale, evidence, owner, deadline and affected claims. Increase
the protocol revision and use a fresh run for new bound outputs. Tests reject a
latest decision that disagrees with the protocol or drops its revision history.

Later lifecycle declarations need a named reviewer, complete bindings and the
relevant agreed topics. Their validation is planning consistency only; the API
always leaves scientific authorization and final confirmatory readiness false.
Before any real fitting, resolve its required decisions and pass the relevant
data/exposure/protocol gates through the complete runner.
