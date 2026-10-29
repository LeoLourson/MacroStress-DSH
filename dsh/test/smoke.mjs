/**
 * The plugin without a running DSH: the skills it registers and the system prompt section
 * it adds. Run with `node dsh/test/smoke.mjs`.
 */

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { apply } from '../index.js';

const here = dirname(fileURLToPath(import.meta.url));
const workspace = join(here, '..', '..', 'workspace');
const skills = [];
const sections = [];

const ctx = {
  effect(effect) {
    effect();
    return () => {};
  },
  skills: {
    register(skill) {
      skills.push(skill);
      return () => {};
    },
  },
  systemPrompt: {
    section(section) {
      sections.push(section);
    },
  },
};

apply(ctx, { workspace });

assert.equal(sections.length, 1, 'the plugin adds one prompt section');
const [section] = sections;
assert.equal(section.name, 'macrostress');
assert.equal(section.order, 1000);
const lines = section.text({}).split('\n');
assert.equal(lines.length, 5, 'the prompt section is five lines');

const state = JSON.parse(readFileSync(join(workspace, 'workspace.json'), 'utf8'));
const short = (value) => (typeof value === 'string' && value !== '' ? value.slice(0, 12) : '—');
const corpus = state.corpus_index?.as_of ?? 'не индексирован';

assert.match(lines[0], /^MacroStress:/);
assert.ok(lines[1].includes(state.workspace), 'the workspace path');
assert.ok(lines[2].includes(short(state.macro)), 'the macro version');
assert.ok(lines[2].includes(short(state.satellites)), 'the satellite version');
assert.ok(lines[2].includes(corpus), 'the corpus index date');
assert.ok(lines[3].includes(`${state.portfolios} портфелей`), 'the portfolio count');
assert.ok(lines[3].includes(`${state.scenarios} сценариев`), 'the scenario count');
assert.ok(lines[3].includes(`${state.runs} прогонов`), 'the run count');
assert.ok(
  lines[3].includes(`задач в работе: ${state.jobs ?? 0}`),
  'the unfinished job count',
);
assert.ok(lines[4].includes('jobs(action="state")'), 'the starting call');

const names = skills.map((skill) => skill.name).sort();
assert.deepEqual(names, ['drivers', 'run-report', 'scenario-from-corpus']);
for (const skill of skills) {
  assert.ok(skill.description.length > 0, `${skill.name}: description`);
  assert.ok(skill.content.length > 0, `${skill.name}: content`);
  assert.equal(skill.source, 'runtime');
}
const content = Object.fromEntries(skills.map((skill) => [skill.name, skill.content]));
assert.ok(content['scenario-from-corpus'].includes('corpus(action="search"'));
assert.ok(content['scenario-from-corpus'].includes('scenario(action="save"'));
assert.ok(content['scenario-from-corpus'].includes('scenario(action="validate"'));
assert.ok(content['run-report'].includes('run(action="result"'));
assert.ok(content['run-report'].includes('P5'));
assert.ok(content['run-report'].includes('ES95'));
assert.ok(content['drivers'].includes('model(kind="satellites"'));
assert.ok(content['drivers'].includes('served'));

for (const locale of ['ru', 'en']) {
  const card = JSON.parse(readFileSync(join(here, '..', 'locale', `${locale}.json`), 'utf8'));
  assert.ok(card.meta.title, `${locale}: title`);
  assert.ok(card.meta.description, `${locale}: description`);
}

console.log(`ok: ${skills.length} skills, ${lines.length} prompt lines, both locales`);
