# Documentation map

Start with the repository [README](../README.md), or open the searchable offline
[HTML guide](index.html). Documentation is grouped by purpose:

For normal use, read [Run and compare experiments](guides/EXPERIMENTS.md),
then the [configuration reference](guides/EXPERIMENT_CONFIG.md). The active
implementation path is [five practical milestones](research/EXPERIMENT_ROADMAP.md).
M0/M1 package specs and gate records are historical/optional infrastructure
references, not an approval workflow required for experiments.

```text
docs/
  README.md                # This map
  index.html               # Generated offline guide
  guides/                  # Setup, commands, data, checkpoints and testing
  research/                # Current milestones and scientific objectives
  specs/m0/M0_SPEC.md       # Historical contract at its runtime-compatible path
  archive/research/        # Superseded protocols and detailed plans
  archive/specs/           # Historical M0/M1 checklists and APIs
  archive/                 # Archive index, earlier repairs and citations
  assets/                  # HTML guide styles and scripts
```

| Need | Start here |
| --- | --- |
| Install and operate the project | [User/developer guides](guides/README.md) |
| Understand the research objective and implementation path | [Research plans](research/README.md) |
| Find historical specifications, protocols or optional gate tools | [Archive/reference index](archive/README.md) |
| Check outstanding full-data experimental acceptance | [Active testing checklist](guides/TESTING.md#full-data-acceptance-checklist) |
| Understand previous repairs or collected source references | [Historical archive](archive/README.md) |

The simple experiment roadmap is the active development plan. The earlier
research roadmap, milestone specifications and archived repair roadmap preserve
design/engineering history. Planned scientific experiments and proposed methods
are not existing results.

Edit Markdown in the appropriate group, then run `python tools/build_docs.py` to
refresh `index.html`. Command examples assume the repository root as working
directory, regardless of the Markdown file's folder. Runtime data/checkpoints
retain their documented locations.
