/**
 * The plugin without a running DSH: the skills it registers and the system prompt section
 * it adds. Run with `node dsh/test/smoke.mjs`.
 *
 * The test builds its own workspace.json in a temporary directory, so it works on a clean
 * checkout where workspace/ is absent. It also checks the one-line fallback, the lazy
 * re-read and that disposing the effect removes the skills.
 */

import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, rmSync, utimesSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { apply, defaultWorkspace } from '../index.js';

const here = dirname(fileURLToPath(import.meta.url));

function context() {
  const skills = [];
  const sections = [];
  const disposers = [];
  const ctx = {
    effect(effect) {
      disposers.push(effect());
      return () => {};
    },
    skills: {
      register(skill) {
        skills.push(skill);
        return () => {
          const at = skills.indexOf(skill);
          if (at >= 0) skills.splice(at, 1);
        };
      },
    },
    systemPrompt: {
      section(section) {
        sections.push(section);
      },
    },
  };
  return { ctx, skills, sections, disposers };
}

function counts(state) {
  return (
    `portfolios ${state.portfolios} · scenarios ${state.scenarios} · runs ${state.runs} · ` +
    `jobs running ${state.jobs ?? 0}`
  );
}

const state = {
  workspace: 'C:\\work\\macrostress',
  macro: 'a323319799bc8ad5ecd51ae29e92b32b2d9f26c50eb1b3e5f9a3170f8b36d996',
  satellites: '3121fdc2eb8bf6088bee32dce0ce3082ed237535522a726071a9847b1ce1941c',
  corpus_index: { as_of: '2026-10-29' },
  portfolios: 1,
  scenarios: 2,
  runs: 3,
  jobs: 4,
};

// The default workspace follows the rule of mst.mcp, and the patch names no machine.
assert.equal(defaultWorkspace({ MACROSTRESS_WORKSPACE: 'D:\\data' }, 'H'), 'D:\\data');
assert.equal(defaultWorkspace({}, 'H'), join('H', '.macrostress', 'workspace'));
const patch = readFileSync(join(here, '..', 'cordis.patch.yml'), 'utf8');
assert.doesNotMatch(patch, /[A-Za-z]:[\\/]|\/home\/|\/Users\//, 'an absolute path in the patch');

const workspace = mkdtempSync(join(tmpdir(), 'dsh-macrostress-'));
try {
  writeFileSync(join(workspace, 'workspace.json'), JSON.stringify(state, null, 2));

  const first = context();
  apply(first.ctx, { workspace });

  assert.equal(first.sections.length, 1, 'the plugin adds one prompt section');
  const [section] = first.sections;
  assert.equal(section.name, 'macrostress');
  assert.equal(section.order, 1000);
  let lines = section.text({}).split('\n');
  assert.equal(lines.length, 5, 'the prompt section is five lines');

  assert.match(lines[0], /^MacroStress:/);
  assert.ok(lines[1].includes(state.workspace), 'the workspace path');
  assert.ok(lines[2].includes(state.macro.slice(0, 12)), 'the macro version');
  assert.ok(lines[2].includes(state.satellites.slice(0, 12)), 'the satellite version');
  assert.ok(lines[2].includes(state.corpus_index.as_of), 'the corpus index date');
  assert.ok(lines[3].includes(counts(state)), 'the counts and unfinished jobs');
  assert.ok(lines[4].includes('jobs(action="state")'), 'the starting call');

  const names = first.skills.map((skill) => skill.name).sort();
  assert.deepEqual(names, ['drivers', 'run-report', 'scenario-from-corpus']);
  for (const skill of first.skills) {
    assert.ok(skill.description.length > 0, `${skill.name}: description`);
    assert.ok(skill.content.length > 0, `${skill.name}: content`);
    assert.equal(skill.source, 'runtime');
  }
  const content = Object.fromEntries(first.skills.map((skill) => [skill.name, skill.content]));
  assert.ok(content['scenario-from-corpus'].includes('corpus(action="search"'));
  assert.ok(content['scenario-from-corpus'].includes('scenario(action="save"'));
  assert.ok(content['scenario-from-corpus'].includes('scenario(action="validate"'));
  assert.ok(content['run-report'].includes('run(action="result"'));
  assert.ok(content['run-report'].includes('P5'));
  assert.ok(content['run-report'].includes('ES95'));
  assert.ok(content['drivers'].includes('model(kind="satellites"'));
  assert.ok(content['drivers'].includes('served'));

  // The section is re-read when workspace.json changes, not cached forever.
  const changed = { ...state, portfolios: 2, scenarios: 1, runs: 0, jobs: 0 };
  writeFileSync(join(workspace, 'workspace.json'), JSON.stringify(changed, null, 2));
  const later = new Date(Date.now() + 60_000);
  utimesSync(join(workspace, 'workspace.json'), later, later);
  lines = section.text({}).split('\n');
  assert.ok(lines[3].includes(counts(changed)), 'the counts are re-read');

  // Disposing the effect removes the skills it registered.
  for (const dispose of first.disposers) dispose();
  assert.equal(first.skills.length, 0, 'skills are disposed with the plugin');
} finally {
  rmSync(workspace, { recursive: true, force: true });
}

// A workspace without workspace.json has a one-line fallback instead of a crash.
const empty = mkdtempSync(join(tmpdir(), 'dsh-macrostress-empty-'));
try {
  const fallback = context();
  apply(fallback.ctx, { workspace: empty });
  const [section] = fallback.sections;
  const text = section.text({});
  assert.equal(text.split('\n').length, 1, 'the missing snapshot is one line');
  assert.match(text, /jobs\(action="state"\)/);
} finally {
  rmSync(empty, { recursive: true, force: true });
}

for (const locale of ['ru', 'en']) {
  const card = JSON.parse(readFileSync(join(here, '..', 'locale', `${locale}.json`), 'utf8'));
  assert.ok(card.meta.title, `${locale}: title`);
  assert.ok(card.meta.description, `${locale}: description`);
}

console.log('ok: 3 skills, 5 prompt lines, fallback, re-read, disposal, both locales');
