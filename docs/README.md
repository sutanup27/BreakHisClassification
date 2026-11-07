# Documentation map

Start with the repository [README](../README.md), or open the searchable offline
[HTML guide](index.html). Documentation is grouped by purpose:

```text
docs/
  README.md                # This map
  index.html               # Generated offline guide
  guides/                  # Setup, commands, data, checkpoints and testing
  research/                # Scientific objectives and execution plans
  specs/m0/                # M0 master contract, tracker and six checklists
  specs/m1/                # M1 master contract, tracker and six checklists
  archive/                 # Historical engineering roadmap and citations
  assets/                  # HTML guide styles and scripts
```

| Need | Start here |
| --- | --- |
| Install and operate the project | [User/developer guides](guides/README.md) |
| Understand the research objective and implementation path | [Research plans](research/README.md) |
| Implement M0 in manageable packages | [M0 tracker](specs/m0/README.md), then [master spec](specs/m0/M0_SPEC.md) |
| Plan and implement the next research infrastructure milestone | [M1 tracker](specs/m1/README.md), then [master spec](specs/m1/M1_SPEC.md) |
| Check outstanding full-data experimental acceptance | [Active testing checklist](guides/TESTING.md#full-data-acceptance-checklist) |
| Understand previous repairs or collected source references | [Historical archive](archive/README.md) |

The research roadmap is the active research plan. The archived repair roadmap is
engineering history. M0's master specification remains the implementation source
of truth; its package checklists track delivery. Planned features are not existing
capabilities.

Edit Markdown in the appropriate group, then run `python tools/build_docs.py` to
refresh `index.html`. Command examples assume the repository root as working
directory, regardless of the Markdown file's folder. Runtime data/checkpoints
retain their documented locations.
