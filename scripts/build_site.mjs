import { cp, mkdir, rm } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const outputDir = path.join(projectRoot, 'dist');
const publicReports = [
  'llama-q4-900-eval-annotated.html',
  'llama-variant-comparison.json',
  'llama-cpu-harness-benchmark.json',
  'model-comparison.md'
];

await rm(outputDir, { recursive: true, force: true });
await mkdir(outputDir, { recursive: true });
await cp(path.join(projectRoot, 'site'), outputDir, { recursive: true });

const reportsDir = path.join(outputDir, 'reports');
await mkdir(reportsDir, { recursive: true });
for (const report of publicReports) {
  await cp(path.join(projectRoot, 'reports', report), path.join(reportsDir, report));
}

console.log(`Built static site in ${path.relative(projectRoot, outputDir)}`);
