// The stacks read the rendered UserData from assets/userdata/<vla>.sh (gitignored, written by
// generate.py / deploy.py). Seed a placeholder for every target the tests synthesize, once for the
// whole run: jest runs the test files in parallel workers, and per-file seeding let one file's
// afterAll delete a placeholder another file was still using. Only placeholders created here are removed.
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

const USERDATA_DIR = path.resolve(__dirname, '../../assets/userdata');
export const SEED_VLAS = ['gr00t', 'pi', 'openarm-lift-act', 'xvla', 'real2sim', 'gr00t-bridge', 'rldx-bridge', 'smolvla',
  'openvla-oft', 'robolab-cosmos3', 'smolvla-robotwin'];
const LEDGER = path.join(os.tmpdir(), `vla-sim-jest-seeded-${process.pid}-${path.basename(path.resolve(__dirname, '../..'))}.json`);

export function setup(): void {
  fs.mkdirSync(USERDATA_DIR, { recursive: true });
  const seeded: string[] = [];
  for (const vla of SEED_VLAS) {
    const p = path.join(USERDATA_DIR, `${vla}.sh`);
    if (!fs.existsSync(p)) {
      fs.writeFileSync(p, '#!/bin/bash\n# test placeholder\n');
      seeded.push(p);
    }
  }
  fs.writeFileSync(LEDGER, JSON.stringify(seeded));
}

export function teardown(): void {
  if (!fs.existsSync(LEDGER)) return;
  for (const p of JSON.parse(fs.readFileSync(LEDGER, 'utf8')) as string[]) fs.rmSync(p, { force: true });
  fs.rmSync(LEDGER, { force: true });
}
