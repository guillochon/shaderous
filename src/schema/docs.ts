// The tables of the docs that restate a table of the code: written by `npm run gen` between
// markers, so that the prose around them stays hand-written.
//
//   <!-- gen:NAME -->
//   ...generated, do not edit...
//   <!-- /gen:NAME -->
import { SECTIONS } from '../../tools/headless/flags.mjs';
import { VIEW_EXTRAS, VIEW_OPTIONS } from '../render/view_options';
import { flatten, OPTIONS, PART_OPTIONS, type Option } from '../setup/options';
import { FLOWS } from '../setup/parts/flow';
import { linkKeys } from '../ui/link';
import { MCP_TOOLS } from '../ui/mcp_schema';

const cell = (text: string) => text.replace(/\|/g, '\\|').replace(/\n/g, ' ');
const table = (head: string[], rows: string[][]) => [`| ${head.join(' | ')} |`, `| ${head.map(() => '---').join(' | ')} |`, ...rows.map((r) => `| ${r.map(cell).join(' | ')} |`)].join('\n');
const code = (v: unknown) => `\`${typeof v === 'string' ? v : JSON.stringify(v)}\``;

/** Every key of a link, in the order a link writes them. */
export function linkKeyTable(): string {
  const groups = { run: 'The run', settings: 'Settings', view: 'The view' };
  return table(['Key', 'Of', 'Meaning'], linkKeys().map((k) => [`\`${k.key}\``, groups[k.group], k.detail ? `${k.about}. ${k.detail}` : k.about]));
}

/** The same keys in three lines, for a reader who needs the names and will ask `describe` for the rest. */
export function linkKeyList(): string {
  const of = (group: string) => linkKeys().filter((k) => k.group === group).map((k) => `\`${k.key}\``).join(', ');
  return [`- The run: ${of('run')}`, `- Settings, each an override of the preset's own: ${of('settings')}`, `- The view: ${of('view')}`].join('\n');
}

/** Every setting `view()` takes, with its key in a link. */
export function viewTable(): string {
  return table(['Key', 'In a link', 'Meaning'], [...VIEW_OPTIONS, ...VIEW_EXTRAS].map((o) => [`\`${o.key}\``, o.link ? `\`${o.link}\`` : '', o.doc]));
}

/** What an option's value may be, and its default. */
function valuesOf(o: Option): string {
  const c = o.control;
  const range = (min?: number, max?: number) => (min !== undefined && max !== undefined ? `${min} to ${max}` : min !== undefined ? `at least ${min}` : max !== undefined ? `at most ${max}` : '');
  const what =
    c.type === 'select' ? c.choices.map((k) => code(k.value)).join(', ')
    : c.type === 'number' ? [c.integer ? 'a whole number' : 'a number', range(c.min, c.max), c.nullable ? 'or `null`' : ''].filter(Boolean).join(', ')
    : c.type === 'check' ? '`true` or `false`'
    : c.type === 'text' ? 'text'
    : c.type === 'vec3' ? '`[x, y, z]`'
    : c.type === 'faces' ? '`{ "lo": [x, y, z], "hi": [x, y, z] }`, each 0 to 3'
    : c.type === 'ref' ? `${c.count === 1 ? 'the id' : c.count === 2 ? 'the ids of two' : 'ids'} of ${c.kinds.map((k) => `\`${k}\``).join(' or ')} parts`
    : c.type === 'fractions' ? '`{ species: fraction }`'
    : c.type === 'names' ? 'a list of names'
    : c.optional ? 'an object, which may be left out' : 'an object';
  const def = o.default === undefined || o.default === null || c.type === 'group' ? '' : `. Default ${code(o.default)}`;
  return what + def;
}

const optionRows = (options: Option[]) => options.map((o) => [`\`${o.path}\``, o.label, valuesOf(o), o.tip]);
const OPTION_HEAD = ['Key', 'In the dialog', 'Values', 'Meaning'];

/** Every general option of a setup document, by the section of the document it is in. */
export function setupOptionTables(): string {
  const all = flatten(OPTIONS);
  // A key of the document itself (its name) goes under the first heading.
  const sectionOf = (o: Option) => (o.path.includes('.') ? o.path.split('.')[0] : '');
  // In the order a document is written; a section this list does not know comes last.
  const order = ['', 'domain', 'physics', 'mesh', 'background', 'run', 'view'];
  const place = (s: string) => (order.includes(s) ? order.indexOf(s) : order.length);
  const sections = [...new Set(all.map(sectionOf))].sort((a, b) => place(a) - place(b));
  return sections.map((s) => `**${s ? `\`${s}\`` : 'The document'}**\n\n${table(OPTION_HEAD, optionRows(all.filter((o) => sectionOf(o) === s)))}`).join('\n\n');
}

/** Every kind of part with its fields. */
export function setupPartTables(): string {
  return Object.values(PART_OPTIONS)
    .map((k) => {
      const head = `**\`${k.kind}\`** (${k.label}). ${k.tip}${k.max !== undefined ? `. At most ${k.max}` : ''}`;
      if (k.kind === 'flow') return `${head}\n\nIts fields are its \`name\` and that flow's parameters, listed below.`;
      return k.fields.length ? `${head}\n\n${table(OPTION_HEAD, optionRows(flatten(k.fields)))}` : head;
    })
    .join('\n\n');
}

/** Every built-in flow with its parameters and their defaults. */
export function setupFlowTable(): string {
  return table(['`name`', 'What it is', 'Parameters, with their defaults'], Object.values(FLOWS).map((f) => [
    `\`${f.name}\``,
    `${f.label}. ${f.tip}${f.wholeBox ? ' (fills the box)' : ''}`,
    Object.entries(f.fields).map(([key, field]) => `\`${key}\` ${field.default} (${field.label})`).join('; '),
  ]));
}

/** Every option of the command-line runner, by the heading its help puts it under. */
export function cliOptionTables(): string {
  return SECTIONS.map((s) => {
    const rows = s.entries.filter((e) => e.flags).map((e) => [e.flags!.map((f) => `\`--${f}\``).join(', '), e.help.join(' ')]);
    return `**${s.title.replace(/\n/g, ' ')}**\n\n${table(['Option', 'Meaning'], rows)}`;
  }).join('\n\n');
}

/** The same flags as one line per heading. */
export function cliFlagList(): string {
  return SECTIONS.map((s) => `- ${s.title.split(/ \(|\n/)[0]}: ${s.entries.flatMap((e) => e.flags ?? []).map((f) => `\`--${f}\``).join(', ')}`).join('\n');
}

/** The MCP server's tools, a line each. */
export function mcpToolTable(): string {
  return table(['Tool', 'What it does'], MCP_TOOLS.map((t) => [`\`${t.name}\``, t.brief]));
}

/** The generated blocks, by name. */
export const DOC_BLOCKS: Record<string, () => string> = {
  'link-keys': linkKeyTable,
  'link-key-list': linkKeyList,
  view: viewTable,
  'setup-options': setupOptionTables,
  'setup-parts': setupPartTables,
  'setup-flows': setupFlowTable,
  'cli-options': cliOptionTables,
  'cli-flag-list': cliFlagList,
  'mcp-tools': mcpToolTable,
};

/** A document with each of its marked blocks filled in. A marker with no block of that name is an error. */
export function fillGenerated(text: string): string {
  return text.replace(/(<!-- gen:([\w-]+) -->\n)[\s\S]*?(<!-- \/gen:\2 -->)/g, (_, open: string, name: string, close: string) => {
    const make = DOC_BLOCKS[name];
    if (!make) throw new Error(`no generated block named "${name}"`);
    return `${open}${make()}\n${close}`;
  });
}
