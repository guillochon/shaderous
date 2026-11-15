// Which of a shader's bindings its entry points use: the bindings a pipeline with layout 'auto'
// has in its layout. A binding is in it if an entry point refers to it, directly or through
// any function it calls (WGSL's static use: whether the code runs does not matter). Found by
// following identifiers from the entry points through function bodies, after comments are
// stripped; enough for WGSL, which has no function pointers and no overloading of user functions.
//
// gpu/device.ts uses it to add a device's shared resources (setSharedBinding: the equation of
// state's table) to exactly the bind groups whose pipelines use them.

const cache = new Map<string, Set<number>>();

function strip(code: string): string {
  return code.replace(/\/\*[\s\S]*?\*\//g, ' ').replace(/\/\/[^\n]*/g, ' ');
}

/** The binding numbers (group 0) that the given entry points of a WGSL module use. */
export function usedBindings(code: string, entryPoints: readonly string[]): Set<number> {
  const key = `${entryPoints.join(',')}\u0000${code}`;
  const hit = cache.get(key);
  if (hit) return hit;
  const src = strip(code);
  // Module-scope resources: @group(0) @binding(n) var<...> NAME
  const vars = new Map<string, number>();
  for (const m of src.matchAll(/@group\(\s*0\s*\)\s*@binding\(\s*(\d+)\s*\)\s*var(?:<[^>]*>)?\s+([A-Za-z_]\w*)/g)) vars.set(m[2], Number(m[1]));
  // Function bodies, by brace matching from each `fn name(`.
  const bodies = new Map<string, string>();
  for (const m of src.matchAll(/\bfn\s+([A-Za-z_]\w*)\s*\(/g)) {
    let i = src.indexOf('{', m.index! + m[0].length);
    if (i < 0) continue;
    const start = i;
    let depth = 0;
    for (; i < src.length; i++) {
      if (src[i] === '{') depth++;
      else if (src[i] === '}' && --depth === 0) break;
    }
    // The signature counts too: a parameter's type cannot name a resource, but keep it simple.
    bodies.set(m[1], src.slice(m.index!, i + 1));
    void start;
  }
  const used = new Set<number>();
  const seen = new Set<string>();
  const queue = [...entryPoints];
  while (queue.length) {
    const f = queue.pop()!;
    if (seen.has(f)) continue;
    seen.add(f);
    const body = bodies.get(f);
    if (!body) continue;
    for (const id of body.matchAll(/[A-Za-z_]\w*/g)) {
      const name = id[0];
      const b = vars.get(name);
      if (b !== undefined) used.add(b);
      if (bodies.has(name) && !seen.has(name)) queue.push(name);
    }
  }
  cache.set(key, used);
  return used;
}
