# dsh/skills

The skills the plugin registers in DSH. Each skill is a `*.md` file with frontmatter:

```markdown
---
name: scenario-from-corpus
description: How to assemble a scenario from the corpus and check its price
whenToUse: optional; when the agent should take this skill
---

The text of the skill.
```

`../index.js` reads all `*.md` files of this folder in alphabetical order and skips files without `name`, `description` or text. The READMEs have no frontmatter, so they do not become skills. The skills themselves are in English because the model reads them, and English costs fewer tokens.

| File | What it covers |
|---|---|
| `scenario-from-corpus.md` | How to turn an event into a scenario $s=(N,m,O,D,\mathcal C)$ from the corpus: the calendar, conditions, impacts, causes, labels and human approval |
| `run-report.md` | How to start a run and read $P_5$/$P_{50}$/$P_{95}$, ES95, the price, the coverage and the flags |
| `drivers.md` | How to update the catalog, estimate satellites and what "satellite rejected" means |

A check without starting DSH is `node dsh/test/smoke.mjs`: it supplies a fake context, registers the skills, builds five section lines from a temporary `workspace.json`, and checks the one-line fallback, the rereading of the file and the removal of the skills.
