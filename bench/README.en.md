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
| [`scenario-from-corpus/`](scenario-from-corpus/events.toml) | how the agent builds a scenario from the text of an event, as the skill `scenario-from-corpus` tells it; 100 events that never happened, so the agent cannot know the outcome | no metric until the expected decisions are added |

## Rules

- A set lives here when it has right answers and a metric. Material the system builds its answer
  from (the corpus, series, scenarios in the workspace) does not go here.
- The inputs of a set and its right answers may live apart. When an agent with access to the
  repository takes the set, the answers are kept outside its working folder: an agent that has
  read the answer makes the check worthless. For `scenario-from-corpus/` the repository holds only
  the texts of the events, and the expected decisions are added separately.
- A new set gets a folder named after the job it measures, and the description of its format,
  metric and command in the head of its file or in the folder's README.

## What stays outside `bench/`

The historical episodes of the macro model check live in
[`bench/model-backtest/episodes.toml`](model-backtest/episodes.toml). By kind it is a check set
too, but `python -m mst.metrics` reads it from the installed package, and the backtest report
keeps the file's hash. Moving it would break the run from the PyPI package and the comparison
with earlier reports, so the file stays where it is.
