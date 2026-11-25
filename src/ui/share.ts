// The Share dialog: the link just copied, as a QR code for a phone to scan.
import qrcode from 'qrcode-generator';

let dialog: HTMLDialogElement | undefined;

/** The link as an SVG QR code, or null when it is too long for one. Medium error correction when
 *  the link fits it, low for a longer link (fewer, larger modules scan more easily on a screen). */
export function qrSvg(text: string): string | null {
  for (const level of ['M', 'L'] as const) {
    try {
      const qr = qrcode(0, level);
      qr.addData(text, 'Byte');
      qr.make();
      return qr.createSvgTag({ cellSize: 1, margin: 4, scalable: true });
    } catch {
      // Too long at this level: try the next.
    }
  }
  return null;
}

function build(): HTMLDialogElement {
  const el = document.createElement('dialog');
  el.className = 'share-dialog';
  el.setAttribute('aria-labelledby', 'share-title');
  el.innerHTML = `
    <button class="share-x" type="button" title="Close" aria-label="Close">×</button>
    <h2 id="share-title"></h2>
    <div class="share-qr"></div>
    <input class="share-url" type="text" readonly aria-label="Share link">`;
  el.querySelector('.share-x')!.addEventListener('click', () => el.close());
  // A click on the backdrop closes it. Both the press and the release must land outside the box,
  // so a selection dragged out of the link field does not dismiss it.
  const outside = (e: MouseEvent) => {
    const r = el.getBoundingClientRect();
    return e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom;
  };
  let pressedOutside = false;
  el.addEventListener('pointerdown', (e) => { pressedOutside = e.target === el && outside(e); });
  el.addEventListener('click', (e) => {
    if (pressedOutside && e.target === el && outside(e)) el.close();
    pressedOutside = false;
  });
  const url = el.querySelector<HTMLInputElement>('.share-url')!;
  url.addEventListener('focus', () => url.select());
  document.body.append(el);
  return el;
}

/** Opens the dialog for `url`; `copied` says whether it reached the clipboard. */
export function showShare(url: string, copied: boolean): void {
  dialog ??= build();
  dialog.querySelector('h2')!.textContent = copied
    ? 'Share link copied to clipboard!'
    : 'Could not reach the clipboard: copy the link below';
  const svg = qrSvg(url);
  const box = dialog.querySelector<HTMLElement>('.share-qr')!;
  if (svg) box.innerHTML = svg;
  else box.innerHTML = '<p>This link is too long for a QR code.</p>';
  box.classList.toggle('none', !svg);
  dialog.querySelector<HTMLInputElement>('.share-url')!.value = url;
  if (!dialog.open) dialog.showModal();
  dialog.querySelector<HTMLElement>('.share-x')!.focus();
}
