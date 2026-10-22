const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

// Catalog cards and detail heroes both use model.coverImage. A root-absolute
// path 404s on GitHub Pages project sites, and an SVG with illegal XML
// control bytes is a broken image even when the file itself returns 200.

function assertWellFormedSvg(text, label) {
  assert(
    !/[\u0000-\u0008\u000B\u000C\u000E-\u001F]/.test(text),
    `${label}: SVG contains illegal XML control characters and renders as a broken image`,
  );
  const stripped = text.replace(/<\?xml[\s\S]*?\?>/g, '').replace(/<!--[\s\S]*?-->/g, '').trim();
  assert(stripped.startsWith('<svg'), `${label}: SVG root is missing`);
  assert(stripped.endsWith('</svg>'), `${label}: SVG is not closed`);
  const tokens = stripped.match(/<\/?[^>]+>/g) || [];
  const stack = [];
  for (const token of tokens) {
    if (token.startsWith('</')) {
      const name = token.slice(2, -1).trim().split(/\s/)[0];
      assert.equal(stack.pop(), name, `${label}: SVG tag mismatch at ${token}`);
    } else if (!token.endsWith('/>')) {
      stack.push(token.slice(1).split(/[\s/>]/)[0]);
    }
  }
  assert.equal(stack.length, 0, `${label}: SVG has unclosed tags`);
}

function assertLocalImageFile(filePath, label) {
  assert(fs.existsSync(filePath), `${label}: cover 404, missing ${filePath}`);
  const bytes = fs.readFileSync(filePath);
  assert(bytes.length > 32, `${label}: cover is empty`);
  const ext = path.extname(filePath).toLowerCase();
  if (ext === '.svg') {
    assertWellFormedSvg(bytes.toString('utf8'), label);
    return;
  }
  const png = bytes[0] === 0x89 && bytes[1] === 0x50 && bytes[2] === 0x4e && bytes[3] === 0x47;
  const jpg = bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff;
  const gif = bytes.subarray(0, 3).toString('ascii') === 'GIF';
  const webp = bytes.subarray(0, 4).toString('ascii') === 'RIFF' && bytes.subarray(8, 12).toString('ascii') === 'WEBP';
  assert(png || jpg || gif || webp, `${label}: cover is not a PNG, JPEG, GIF, WebP, or SVG`);
}

function assertCoverResolves(coverImage, distRoot, label) {
  assert(coverImage, `${label}: catalog cover and detail hero image are missing`);
  assert(
    !coverImage.startsWith('/'),
    `${label}: cover ${coverImage} is root-absolute and 404s under a GitHub Pages project path`,
  );
  assert(!/^[a-z][a-z0-9+.-]*:/i.test(coverImage), `${label}: cover must be a local asset: ${coverImage}`);
  const relative = path.relative(distRoot, path.resolve(distRoot, coverImage));
  assert(relative && !relative.startsWith('..') && !path.isAbsolute(relative), `${label}: cover must stay inside the site`);
  assertLocalImageFile(path.join(distRoot, coverImage), `${label}: ${coverImage}`);
}

module.exports = { assertCoverResolves, assertLocalImageFile, assertWellFormedSvg };
