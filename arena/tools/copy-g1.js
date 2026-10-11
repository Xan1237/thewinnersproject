// Copies the generated Unitree G1 model from ../robot (built by `npm run assets` there) into public/g1.
import { execFileSync } from 'node:child_process';
import { cpSync, existsSync, mkdirSync, renameSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const src = join(here, '..', '..', 'robot', 'public', 'assets', 'g1');
const dst = join(here, '..', 'public', 'g1');
if (!existsSync(join(src, 'g1.glb'))) {
  console.error('G1 model missing. Run: (cd ../robot && npm ci && npm run assets)');
  process.exit(1);
}
mkdirSync(dst, { recursive: true });
for (const f of ['g1.glb', 'metadata.json', 'LICENSE-UNITREE.txt', 'PROVENANCE.md']) cpSync(join(src, f), join(dst, f));
// The source model has ~630k triangles; the arena shows several robots at once, so decimate it (~160k).
const full = join(dst, 'g1.glb'), small = join(dst, 'g1.small.glb');
execFileSync('npx', ['--yes', '@gltf-transform/cli@4', 'simplify', full, small, '--ratio', '0.15', '--error', '0.001', '--lock-border', 'false'], { stdio: 'inherit' });
renameSync(small, full);
console.log('copied and simplified G1 assets in', dst);
