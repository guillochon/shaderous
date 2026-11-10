// The error panel: one scrollable, selectable report with a copy button, instead of raw text
// laid over the page. Everything that goes wrong (exceptions, WebGPU validation and shader
// compilation errors) is appended to the same report, with the build and browser it came from.

declare const __BUILD__: string;
export const BUILD: string = typeof __BUILD__ === 'string' ? __BUILD__ : 'dev';

const MAX_SHOWN = 400; // lines displayed; the copy has everything
const MAX_COPY = 200_000; // characters

const entries: string[] = [];
let adapterInfo = '';

export function setAdapterInfo(text: string) {
  adapterInfo = text;
}

/** Collapse runs of identical lines (shader compilers repeat their notes many times over). */
function collapse(text: string): string[] {
  const out: string[] = [];
  let last = '';
  let run = 0;
  const flush = () => {
    if (run > 1) out[out.length - 1] += `   (x${run})`;
  };
  for (const line of text.split('\n')) {
    if (line === last && line.trim() !== '') {
      run++;
      continue;
    }
    flush();
    out.push(line);
    last = line;
    run = 1;
  }
  flush();
  return out;
}

function fullReport(): string {
  const head = [
    `Shaderous build ${BUILD}`,
    `Page: ${location.href}`,
    `Browser: ${navigator.userAgent}`,
    adapterInfo ? `GPU: ${adapterInfo}` : '',
    `Errors: ${entries.length}`,
  ].filter(Boolean);
  const body = entries.map((e, i) => `--- ${i + 1} ---\n${collapse(e).join('\n')}`).join('\n\n');
  return (head.join('\n') + '\n\n' + body).slice(0, MAX_COPY);
}

function render() {
  const box = document.getElementById('error');
  if (!box) return;
  box.style.display = 'flex';
  if (getComputedStyle(box).position === 'fixed') {
    // Full-screen on a phone: keep to the page's own width, whatever the browser calls the viewport.
    box.style.right = 'auto';
    box.style.bottom = 'auto';
    box.style.width = document.documentElement.clientWidth + 'px';
    box.style.height = document.documentElement.clientHeight + 'px';
  }
  box.innerHTML = `
    <div class="err-head">
      <div>
        <div class="err-title"></div>
        <div class="err-sub"></div>
      </div>
      <div class="err-actions">
        <button id="err-copy">Copy details</button>
        <button id="err-reload">Reload</button>
      </div>
    </div>
    <pre class="err-body" tabindex="0"></pre>`;
  const first = entries[0].split('\n').find((l) => l.trim()) ?? '';
  box.querySelector('.err-title')!.textContent = entries.length === 1 ? 'Something went wrong' : `${entries.length} errors`;
  box.querySelector('.err-sub')!.textContent = `${first.slice(0, 160)}${first.length > 160 ? '…' : ''}  ·  build ${BUILD}`;
  const lines = fullReport().split('\n');
  const body = box.querySelector<HTMLPreElement>('.err-body')!;
  body.textContent =
    lines.length > MAX_SHOWN ? lines.slice(0, MAX_SHOWN).join('\n') + `\n\n… ${lines.length - MAX_SHOWN} more lines; "Copy details" has all of it.` : lines.join('\n');

  const copy = box.querySelector<HTMLButtonElement>('#err-copy')!;
  copy.onclick = async () => {
    try {
      await navigator.clipboard.writeText(fullReport());
      copy.textContent = 'Copied';
    } catch {
      // No clipboard access: select the text so it can be copied by hand.
      const range = document.createRange();
      range.selectNodeContents(body);
      const sel = getSelection();
      sel?.removeAllRanges();
      sel?.addRange(range);
      copy.textContent = 'Selected: copy it';
    }
    setTimeout(() => (copy.textContent = 'Copy details'), 2500);
  };
  box.querySelector<HTMLButtonElement>('#err-reload')!.onclick = () => location.reload();
}

/** Add an error to the report and show it. Repeats of the same message are counted once. */
export function reportError(e: unknown) {
  const text = e instanceof Error ? e.message + (e.stack && !e.stack.includes(e.message) ? '\n' + e.stack : '') : String(e);
  if (entries.includes(text)) return;
  if (entries.length < 50) entries.push(text);
  console.error(e);
  render();
}

// WebGPU errors nobody caught (validation, out of memory, shader compilation in some browsers).
addEventListener('shaderous-gpu-error', (e) => reportError((e as CustomEvent<string>).detail));
addEventListener('unhandledrejection', (e) => reportError(e.reason));
addEventListener('error', (e) => reportError(e.error ?? e.message));
