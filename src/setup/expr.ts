// Formulas in a setup document: a density profile given as a function, such as
// "sigma0 * (r / rIn)^-1.5 * exp(-(r / rOut)^2)".
//
// A formula is text from a file, and a file may come from a stranger, so it is never run as
// code. It is parsed here into a tree of numbers, named variables, arithmetic and a fixed list
// of functions, and anything else is an error. The tree is then walked: to a number, on the
// CPU, or to WGSL text made only of what this file writes (float literals, the operators, the
// listed functions and the variables' own expressions), for a shader.

type Node =
  | { k: 'num'; v: number }
  | { k: 'var'; name: string }
  | { k: 'neg'; a: Node }
  | { k: 'bin'; op: '+' | '-' | '*' | '/' | '^'; a: Node; b: Node }
  | { k: 'call'; fn: string; args: Node[] };

interface Fn {
  /** how many arguments it takes */
  n: number;
  js: (...a: number[]) => number;
  wgsl: (...a: string[]) => string;
  /** what it is, for the list the dialog shows */
  tip: string;
}

/** The functions a formula may call. */
export const FUNCTIONS: Record<string, Fn> = {
  exp: { n: 1, js: Math.exp, wgsl: (a) => `exp(${a})`, tip: 'e to the power x' },
  log: { n: 1, js: Math.log, wgsl: (a) => `log(${a})`, tip: 'natural logarithm' },
  ln: { n: 1, js: Math.log, wgsl: (a) => `log(${a})`, tip: 'natural logarithm' },
  log10: { n: 1, js: Math.log10, wgsl: (a) => `(log(${a}) * 0.4342944819)`, tip: 'logarithm to base 10' },
  sqrt: { n: 1, js: Math.sqrt, wgsl: (a) => `sqrt(${a})`, tip: 'square root' },
  abs: { n: 1, js: Math.abs, wgsl: (a) => `abs(${a})`, tip: 'absolute value' },
  sin: { n: 1, js: Math.sin, wgsl: (a) => `sin(${a})`, tip: 'sine' },
  cos: { n: 1, js: Math.cos, wgsl: (a) => `cos(${a})`, tip: 'cosine' },
  tan: { n: 1, js: Math.tan, wgsl: (a) => `tan(${a})`, tip: 'tangent' },
  atan: { n: 1, js: Math.atan, wgsl: (a) => `atan(${a})`, tip: 'arctangent' },
  sinh: { n: 1, js: Math.sinh, wgsl: (a) => `sinh(${a})`, tip: 'hyperbolic sine' },
  cosh: { n: 1, js: Math.cosh, wgsl: (a) => `cosh(${a})`, tip: 'hyperbolic cosine' },
  tanh: { n: 1, js: Math.tanh, wgsl: (a) => `tanh(${a})`, tip: 'hyperbolic tangent: a smooth step' },
  step: { n: 1, js: (x) => (x > 0 ? 1 : 0), wgsl: (a) => `select(0.0, 1.0, ${a} > 0.0)`, tip: '1 where x is positive, 0 elsewhere: a sharp edge' },
  pow: { n: 2, js: Math.pow, wgsl: (a, b) => `pow(${a}, ${b})`, tip: 'x to the power y (as x^y)' },
  min: { n: 2, js: Math.min, wgsl: (a, b) => `min(${a}, ${b})`, tip: 'the smaller of two' },
  max: { n: 2, js: Math.max, wgsl: (a, b) => `max(${a}, ${b})`, tip: 'the larger of two' },
  clamp: { n: 3, js: (x, lo, hi) => Math.min(hi, Math.max(lo, x)), wgsl: (x, lo, hi) => `clamp(${x}, ${lo}, ${hi})`, tip: 'x held between two values' },
};

/** Names every formula knows. */
const CONSTANTS: Record<string, number> = { pi: Math.PI, e: Math.E };

const MAX_LENGTH = 500;
const MAX_NODES = 400;
const MAX_DEPTH = 48;

/** A number as a WGSL f32 literal. */
const lit = (v: number): string => {
  if (!Number.isFinite(v)) throw new Error('a number in the formula is not finite');
  const s = Number.isInteger(v) && Math.abs(v) < 1e15 ? `${v}.0` : String(v);
  return v < 0 ? `(${s})` : s;
};

export interface Formula {
  /** the text it was parsed from */
  text: string;
  /** its value for these variables (NaN where it has none: a logarithm of a negative number) */
  at(vars: Record<string, number>): number;
  /** the same as WGSL, with each variable written as the expression given for it */
  wgsl(vars: Record<string, string>): string;
}

/**
 * Parse a formula over the named variables (constants are given as variables with numbers in
 * `at` and literals in `wgsl`). Throws with a sentence saying what is wrong and where.
 */
export function parseFormula(text: string, variables: readonly string[]): Formula {
  if (typeof text !== 'string') throw new Error('a formula is text');
  if (text.length > MAX_LENGTH) throw new Error(`the formula is longer than ${MAX_LENGTH} characters`);
  const src = text;
  let i = 0;
  let nodes = 0;
  const known = new Set(variables);
  const fail = (what: string): never => {
    throw new Error(`${what} at character ${i + 1} of "${src.length > 60 ? `${src.slice(0, 57)}...` : src}"`);
  };
  const skip = () => {
    while (i < src.length && /\s/.test(src[i])) i++;
  };
  const made = <T extends Node>(n: T): T => {
    if (++nodes > MAX_NODES) fail('the formula is too long');
    return n;
  };
  const eat = (c: string): boolean => {
    skip();
    if (src[i] !== c) return false;
    i++;
    return true;
  };

  function primary(depth: number): Node {
    if (depth > MAX_DEPTH) fail('the formula is nested too deep');
    skip();
    const num = /^(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?/.exec(src.slice(i));
    if (num) {
      i += num[0].length;
      return made({ k: 'num', v: Number(num[0]) });
    }
    const id = /^[A-Za-z_][A-Za-z0-9_]*/.exec(src.slice(i));
    if (id) {
      const name = id[0];
      const at = i;
      i += name.length;
      if (eat('(')) {
        const fn = Object.prototype.hasOwnProperty.call(FUNCTIONS, name) ? FUNCTIONS[name] : undefined;
        if (!fn) {
          i = at;
          fail(`"${name}" is not a function a formula can use (${Object.keys(FUNCTIONS).join(', ')})`);
        }
        const args: Node[] = [];
        if (!eat(')')) {
          do args.push(sum(depth + 1));
          while (eat(','));
          if (!eat(')')) fail('a ")" is missing');
        }
        if (args.length !== fn!.n) {
          i = at;
          fail(`${name} takes ${fn!.n} value${fn!.n > 1 ? 's' : ''}, not ${args.length},`);
        }
        return made({ k: 'call', fn: name, args });
      }
      if (Object.prototype.hasOwnProperty.call(CONSTANTS, name) && !known.has(name)) return made({ k: 'num', v: CONSTANTS[name] });
      if (!known.has(name)) {
        i = at;
        fail(`"${name}" is not a name this formula knows (${[...variables, ...Object.keys(CONSTANTS)].join(', ')})`);
      }
      return made({ k: 'var', name });
    }
    if (eat('(')) {
      const inner = sum(depth + 1);
      if (!eat(')')) fail('a ")" is missing');
      return inner;
    }
    return fail(i >= src.length ? 'the formula ends too soon' : `"${src[i]}" is not expected`);
  }
  // A power binds tighter than a sign before it and takes a signed exponent: -r^-2 is -(r^(-2)).
  function power(depth: number): Node {
    const base = primary(depth);
    if (eat('^')) return made({ k: 'bin', op: '^', a: base, b: unary(depth + 1) });
    // The other spelling of a power.
    skip();
    if (src.startsWith('**', i)) {
      i += 2;
      return made({ k: 'bin', op: '^', a: base, b: unary(depth + 1) });
    }
    return base;
  }
  function unary(depth: number): Node {
    if (depth > MAX_DEPTH) fail('the formula is nested too deep');
    if (eat('-')) return made({ k: 'neg', a: unary(depth + 1) });
    if (eat('+')) return unary(depth + 1);
    return power(depth);
  }
  function product(depth: number): Node {
    let a = unary(depth);
    for (;;) {
      skip();
      if (src[i] === '*' && src[i + 1] !== '*') {
        i++;
        a = made({ k: 'bin', op: '*', a, b: unary(depth) });
      } else if (eat('/')) a = made({ k: 'bin', op: '/', a, b: unary(depth) });
      else return a;
    }
  }
  function sum(depth: number): Node {
    let a = product(depth);
    for (;;) {
      if (eat('+')) a = made({ k: 'bin', op: '+', a, b: product(depth) });
      else if (eat('-')) a = made({ k: 'bin', op: '-', a, b: product(depth) });
      else return a;
    }
  }

  const tree = sum(0);
  skip();
  if (i < src.length) fail(`"${src[i]}" is not expected`);

  const value = (n: Node, vars: Record<string, number>): number => {
    switch (n.k) {
      case 'num': return n.v;
      case 'var': return vars[n.name] ?? NaN;
      case 'neg': return -value(n.a, vars);
      case 'call': return FUNCTIONS[n.fn].js(...n.args.map((a) => value(a, vars)));
      case 'bin': {
        const [a, b] = [value(n.a, vars), value(n.b, vars)];
        return n.op === '+' ? a + b : n.op === '-' ? a - b : n.op === '*' ? a * b : n.op === '/' ? a / b : Math.pow(a, b);
      }
    }
  };
  const shader = (n: Node, vars: Record<string, string>): string => {
    switch (n.k) {
      case 'num': return lit(n.v);
      case 'var': {
        if (!Object.prototype.hasOwnProperty.call(vars, n.name)) throw new Error(`no shader expression for "${n.name}"`);
        return `(${vars[n.name]})`;
      }
      case 'neg': return `(-${shader(n.a, vars)})`;
      case 'call': return FUNCTIONS[n.fn].wgsl(...n.args.map((a) => shader(a, vars)));
      case 'bin': {
        const [a, b] = [shader(n.a, vars), shader(n.b, vars)];
        return n.op === '^' ? `pow(${a}, ${b})` : `(${a} ${n.op} ${b})`;
      }
    }
  };
  return { text: src, at: (vars) => value(tree, vars), wgsl: (vars) => shader(tree, vars) };
}
