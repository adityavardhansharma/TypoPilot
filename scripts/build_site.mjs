import { cp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
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

// Compact per-example outcomes for the interactive 900-square ledger on the home page.
const CATEGORY_CODES = {
  english_extreme: 'EXT', english_heavy: 'HVY', english_medium: 'MED', english_light: 'LGT',
  real_errors_c4: 'C4', no_change_needed: 'NOC', paragraphs_multi: 'PAR'
};
const report = await readFile(path.join(projectRoot, 'reports', 'llama-q4-900-eval-annotated.html'), 'utf8');
const suite = JSON.parse(report.match(/const DATA=(\[.*?\]);/s)[1]);
const outcomeOf = (row) => {
  if (row.exact_match) return 'exact';
  if (row.overcorrected) return 'over';
  if (row.improvement > 0) return 'improved';
  return 'other';
};
const order = ['exact', 'improved', 'over', 'other'];
const clip = (text) => (text.length > 180 ? `${text.slice(0, 177)}…` : text);
const outcomes = suite
  .map((row) => {
    const [category, number] = [row.id.slice(0, row.id.lastIndexOf('-')), row.id.slice(row.id.lastIndexOf('-') + 1)];
    return [`${CATEGORY_CODES[category] ?? category}-${number}`, outcomeOf(row), Math.round(row.latency_ms), clip(row.input), clip(row.hypothesis)];
  })
  .sort((a, b) => order.indexOf(a[1]) - order.indexOf(b[1]));
await mkdir(path.join(outputDir, 'data'), { recursive: true });
await writeFile(path.join(outputDir, 'data', 'outcomes.json'), JSON.stringify(outcomes));

console.log(`Built static site in ${path.relative(projectRoot, outputDir)}`);
