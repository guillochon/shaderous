// The tables of the docs that restate a table of the code: written by `npm run gen` between
// markers, so that the prose around them stays hand-written.
//
//   <!-- gen:NAME -->
//   ...generated, do not edit...
//   <!-- /gen:NAME -->
import { VIEW_EXTRAS, VIEW_OPTIONS } from '../render/view_options';
import { linkKeys } from '../ui/link';

const cell = (text: string) => text.replace(/\|/g, '\|').replace(/\n/g, ' ');
const table = (head: string[], rows: string[][]) => [`| ${head.join(' | ')} |`, `| ${head.map(() => '---').join(' | ')} |`, ...rows.map((r) => `| ${r.map(cell).join(' | ')} |`)].join('\n');

/** Every key of a link, in the order a link writes them. */
export function linkKeyTable(): string {
  const groups = { run: 'The run', settings: 'Settings', view: 'The view' };
  return table(['Key', 'Of', 'Meaning'], linkKeys().map((k) => [`\`${k.key}\``, groups[k.group], k.detail ? `${k.about}. ${k.detail}` : k.about]));
}

/** Every setting `view()` takes, with its key in a link. */
export function viewTable(): string {
  return table(['Key', 'In a link', 'Meaning'], [...VIEW_OPTIONS, ...VIEW_EXTRAS].map((o) => [`\`${o.key}\``, o.link ? `\`${o.link}\`` : '', o.doc]));
}

/** The generated blocks, by name. */
export const DOC_BLOCKS: Record<string, () => string> = {
  'link-keys': linkKeyTable,
  view: viewTable,
};

/** A document with each of its marked blocks filled in. A marker with no block of that name is an error. */
export function fillGenerated(text: string): string {
  return text.replace(/(<!-- gen:([\w-]+) -->\n)[\s\S]*?(<!-- \/gen:\2 -->)/g, (_, open: string, name: string, close: string) => {
    const make = DOC_BLOCKS[name];
    if (!make) throw new Error(`no generated block named "${name}"`);
    return `${open}${make()}\n${close}`;
  });
}
