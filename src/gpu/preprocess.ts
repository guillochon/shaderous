// Minimal WGSL preprocessor.
//
//   #include "path/file.wgsl"   pulls in a file from the shader registry (once per assembly)
//   #include <slot>             pulls in whatever the caller mapped to that slot; the mapping is
//                               either a registry path or literal WGSL text (anything with a newline
//                               or a ';' is treated as text)
//
// Slots are how the equation of state, Riemann solver and limiter are swapped.

export type Slots = Record<string, string>;

const INCLUDE = /^[ \t]*#include[ \t]+(?:"([^"]+)"|<([^>]+)>)[ \t]*$/gm;

export function preprocess(
  entry: string,
  files: Record<string, string>,
  slots: Slots = {},
): string {
  const seen = new Set<string>();
  const stack: string[] = [];

  const expandText = (text: string, label: string): string => {
    if (stack.includes(label)) {
      throw new Error(`#include cycle: ${[...stack, label].join(' -> ')}`);
    }
    stack.push(label);
    const out = text.replace(INCLUDE, (_m, path: string | undefined, slot: string | undefined) => {
      if (path !== undefined) return expandFile(path);
      const target = slots[slot!];
      if (target === undefined) throw new Error(`no mapping for slot <${slot}> (in ${label})`);
      if (target.includes('\n') || target.includes(';')) return expandText(target, `<${slot}>`);
      return expandFile(target);
    });
    stack.pop();
    return out;
  };

  const expandFile = (path: string): string => {
    if (seen.has(path)) return `// (already included: ${path})`;
    const src = files[path];
    if (src === undefined) throw new Error(`shader file not found: ${path}`);
    seen.add(path);
    return `// ---- ${path}\n${expandText(src, path)}\n// ---- end ${path}`;
  };

  return expandFile(entry);
}
