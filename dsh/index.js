/**
 * MacroStress as one DSH plugin: the MCP server that owns a workspace, the skills
 * that describe how to work with it, and the system prompt section that tells every
 * agent which model versions a run would use.
 *
 * The workspace path is the one `cordis.patch.yml` passes to the server, so the
 * section and the tools always describe the same directory.
 */

import { readdirSync, readFileSync, statSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

export const name = 'dsh-macrostress';
export const inject = ['skills', 'systemPrompt'];

const here = dirname(fileURLToPath(import.meta.url));
const defaultWorkspace = join(here, '..', 'workspace');

/** Split one SKILL.md into its frontmatter fields and its body. */
function parseSkill(text) {
  const match = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?/.exec(text);
  const fields = {};
  if (match) {
    for (const line of match[1].split(/\r?\n/)) {
      const at = line.indexOf(':');
      if (at <= 0) continue;
      let value = line.slice(at + 1).trim();
      if (/^(['"]).*\1$/.test(value)) value = value.slice(1, -1);
      fields[line.slice(0, at).trim()] = value;
    }
  }
  return { fields, body: (match === null ? text : text.slice(match[0].length)).trim() };
}

/** Read every bundled skill, skipping files whose frontmatter cannot name one. */
function loadSkills(directory) {
  let files;
  try {
    files = readdirSync(directory).filter((file) => file.endsWith('.md')).sort();
  } catch {
    return [];
  }
  return files
    .map((file) => parseSkill(readFileSync(join(directory, file), 'utf8')))
    .map(({ fields, body }) => ({
      name: fields.name,
      description: fields.description,
      ...fields.whenToUse === undefined ? {} : { whenToUse: fields.whenToUse },
      content: body,
    }))
    .filter((skill) => typeof skill.name === 'string' && skill.description && skill.content);
}

/** The first characters of a version hash, or a dash when there is no version yet. */
function shortVersion(value) {
  return typeof value === 'string' && value !== '' ? value.slice(0, 12) : '—';
}

/** The section text: five lines, whatever the workspace holds. */
function describe(state) {
  const corpus = state.corpus_index?.as_of ?? 'не индексирован';
  return [
    'MacroStress: стресс-тест портфеля, данные лежат в workspace ниже.',
    `workspace ${state.workspace}`,
    `макромодель ${shortVersion(state.macro)} · сателлиты ${shortVersion(state.satellites)} · корпус ${corpus}`,
    `${state.portfolios} портфелей · ${state.scenarios} сценариев · ${state.runs} прогонов · ${state.jobs ?? 0} задач`,
    'Начать с jobs(action="state"): он обновляет этот снимок из файла.',
  ].join('\n');
}

/** Read workspace.json, re-reading only when the file changed. */
function snapshotText(workspace, cache) {
  const file = join(workspace, 'workspace.json');
  let stamp;
  try {
    const info = statSync(file);
    stamp = `${info.mtimeMs}:${info.size}`;
  } catch {
    return `MacroStress: у workspace ${workspace} ещё нет состояния — вызови jobs(action="state").`;
  }
  if (cache.stamp === stamp) return cache.text;
  cache.text = describe(JSON.parse(readFileSync(file, 'utf8')));
  cache.stamp = stamp;
  return cache.text;
}

export function apply(ctx, config = {}) {
  const workspace = config.workspace ?? defaultWorkspace;
  const cache = { stamp: '', text: '' };

  // The registry owns the effect when `register` is called on its own context, so
  // the disposers are chained to this plugin's lifetime here: disabling the plugin
  // removes its skills instead of leaking them.
  ctx.effect(() => {
    const disposers = loadSkills(join(here, 'skills')).map((skill) =>
      ctx.skills.register({ ...skill, source: 'runtime' }));
    return () => {
      for (const dispose of disposers) dispose();
    };
  }, 'dsh-macrostress: skills');

  ctx.systemPrompt.section({
    name: 'macrostress',
    order: 1000,
    text: () => snapshotText(workspace, cache),
  });
}
