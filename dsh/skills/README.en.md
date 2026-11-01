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

`../index.js` reads all `*.md` files of this folder in alphabetical order and skips files without `name`, `description` or text. The READMEs have no frontmatter, so they do not become skills. The skills themselves are in Russian: they are instructions the agent loads at run time, not documentation.

| File | What it covers |
|---|---|
| `scenario-from-corpus.md` | How to search corpus fragments and assemble a scenario; how `save` differs from `validate` |
| `run-report.md` | How to start a run and read $P_5$/$P_{50}$/$P_{95}$, ES95, the price, the coverage and the flags |
| `drivers.md` | How to update the catalog, estimate satellites and what "satellite rejected" means |

A check without starting DSH is `node dsh/test/smoke.mjs`: it supplies a fake context, registers the skills, builds five section lines from a temporary `workspace.json`, and checks the one-line fallback, the rereading of the file and the removal of the skills.
