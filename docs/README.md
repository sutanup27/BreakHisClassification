# Documentation map

Start with the repository [README](../README.md), or open the searchable offline
[HTML guide](index.html). Documentation is grouped by purpose:

For normal use, read [Run and compare experiments](guides/EXPERIMENTS.md),
then the [configuration reference](guides/EXPERIMENT_CONFIG.md). The active
implementation path is [five practical milestones](research/EXPERIMENT_ROADMAP.md).

```text
docs/
  README.md                # This map
  index.html               # Generated offline guide
  guides/                  # Setup, commands, data, checkpoints and testing
  research/                # Current milestones and scientific objectives
  assets/                  # HTML guide styles and scripts
```

| Need | Start here |
| --- | --- |
| Install and operate the project | [User/developer guides](guides/README.md) |
| Understand the research objective and implementation path | [Research plans](research/README.md) |
| Check outstanding full-data experimental acceptance | [Active testing checklist](guides/TESTING.md#full-data-acceptance-checklist) |

The experiment roadmap is the active development plan. The research guide defines
the scientific questions. Planned experiments and proposed methods are not results.

Edit Markdown in the appropriate group, then run `python tools/build_docs.py` to
refresh `index.html`. Command examples assume the repository root as working
directory, regardless of the Markdown file's folder. Runtime data/checkpoints
retain their documented locations.
