import puppeteer from 'puppeteer';
import { readdir } from 'fs/promises';
import { join, basename } from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const SRC_DIR = join(__dirname, 'figures', 'src');
const OUT_DIR = join(__dirname, 'figures');

// Figure configurations - customize dimensions per figure
const FIGURE_CONFIG = {
  default: { width: 1200, height: 800, scale: 2 },
  // Add custom configs per figure name if needed:
  // 'architecture': { width: 1400, height: 600, scale: 2 },
};

async function exportFigure(browser, htmlFile) {
  const name = basename(htmlFile, '.html');
  const config = FIGURE_CONFIG[name] || FIGURE_CONFIG.default;

  const page = await browser.newPage();
  await page.setViewport({
    width: config.width,
    height: config.height,
    deviceScaleFactor: config.scale,
  });

  const filePath = join(SRC_DIR, htmlFile);
  await page.goto(`file://${filePath}`, { waitUntil: 'networkidle0' });

  // Wait for any JavaScript rendering (requestAnimationFrame, etc.)
  await page.evaluate(() => new Promise(r => setTimeout(r, 100)));

  // Get actual content bounds from body (with inline-block, body shrinks to content)
  const clip = await page.evaluate(() => {
    const body = document.body;
    const rect = body.getBoundingClientRect();
    // Account for body padding/margin
    const style = getComputedStyle(body);
    const marginLeft = parseFloat(style.marginLeft) || 0;
    const marginTop = parseFloat(style.marginTop) || 0;
    return {
      x: 0,
      y: 0,
      width: Math.ceil(rect.width + marginLeft * 2),
      height: Math.ceil(rect.height + marginTop * 2),
    };
  });

  const outPath = join(OUT_DIR, `${name}.png`);
  await page.screenshot({
    path: outPath,
    clip: {
      x: 0,
      y: 0,
      width: clip.width,
      height: clip.height,
    },
    omitBackground: false,
  });

  console.log(`✓ ${name}.html → ${name}.png (${clip.width}x${clip.height})`);
  await page.close();
}

async function main() {
  const files = await readdir(SRC_DIR);
  const htmlFiles = files.filter(f => f.endsWith('.html'));

  if (htmlFiles.length === 0) {
    console.log('No HTML files found in figures/src/');
    return;
  }

  console.log(`Exporting ${htmlFiles.length} figure(s)...\n`);

  const browser = await puppeteer.launch();

  for (const file of htmlFiles) {
    await exportFigure(browser, file);
  }

  await browser.close();
  console.log('\nDone!');
}

main().catch(console.error);
