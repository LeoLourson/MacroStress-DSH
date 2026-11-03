# bench

Check sets: tasks with a known right answer that measure the quality of one job of the system.
The tests in [`tests/`](../tests/README.en.md) check that the code does what it should and answer
yes or no, while a set gives a number that is compared between versions of the
search, the models or the agent's instructions.

The folders split on one basis: which job of the system the set measures. A folder is named
after that job.

| Folder | What it measures | Metric and command |
|---|---|---|
| [`corpus-search/`](corpus-search/questions.md) | the corpus search: does it find the fragments that hold the answer to a question asked on the day `as_of` | recall@5, `python -m mst.corpus eval --corpus workspace/corpus --questions bench/corpus-search/questions.md` |
| [`scenario-from-corpus/`](scenario-from-corpus/events.toml) | how the agent builds a scenario from the text of an event, as the skill `scenario-from-corpus` tells it; 100 events that never happened, so the agent cannot know the outcome | the shares of the expert's verdicts match, partial, mismatch; the agent builds the scenario of one event through MCP: `scenario(action="bench", event="E017")`, then `scenario(action="bench_save", ...)`; the summary is `python -m mst.bench scenarios` |
| [`model-backtest/`](model-backtest/episodes.toml) | the quality of the macro models and the satellites on history: episodes R13–R32 with the scenario of what happened, the date grid `[rolling]` and the family of events for the price; one plan for all models | normalized CRPS, coverage, admission and price against the random walk, `python -m mst.metrics backtest MODEL_ID --episodes bench/model-backtest/episodes.toml ...` and `rolling` with the same `--episodes` |

## Rules

- A set lives here when it has right answers and a metric. Material the system builds its answer
  from (the corpus, series, scenarios in the workspace) does not go here.
- The inputs of a set and its right answers may live apart. When an agent with access to the
  repository takes the set, the answers are kept outside its working folder: an agent that has
  read the answer makes the check worthless. For `scenario-from-corpus/` the repository holds only
  the texts of the events, and the expected decisions are added separately.
- A new set gets a folder named after the job it measures, and the description of its format,
  metric and command in the head of its file or in the folder's README.
- An item of a set does not copy the material of the system. An episode of `model-backtest/`
  keeps its start month `start` and links to its corpus card with `card`; the test
  [`tests/test_bench.py`](../tests/test_bench.py) holds `start` equal to the card's start month.

## Results

The result of a set lives beside it, in the `results/` of its folder, and is committed to the
repository. The last result of each model or search is kept, and the earlier ones stay in the
git history. A calculation saves nothing without an explicit command, so trial calculations do
not reach the folder.

| Set | How to save | What is saved |
|---|---|---|
| `model-backtest/` | `backtest ... --save`, over the whole plan only, without `--only` | `results/<model>.json`, the full report that `--against` can name; `results/<model>.md`, the tables the command prints |
| `corpus-search/` | `eval ... --save` | `results/recall.json` and `results/recall.md`: the questions, the marked-up ones, recall@5 |
| `scenario-from-corpus/` | the agent calls `scenario(action="bench_save", event=..., spec=..., notes=...)` | `results/<event>/scenario.json`, the scenario; `journal.json`, the text of the event, every `corpus` call with its answer, and the agent's `notes`; `review.toml`, empty `verdict` and `comment` fields the expert fills by hand after comparing the scenario with the expected decision |

To build the scenario of an event, ask the agent in a session with MCP: "build the bench
scenario for event E017". The agent receives only the text of the event. The journal runs from
the `bench` call to `bench_save`. A bench scenario is not approved (`approved_by` stays empty)
and not computed. Saving an event again replaces its scenario, journal and review.
`python -m mst.bench scenarios` writes the shares of the verdicts to `results/summary.md`.

