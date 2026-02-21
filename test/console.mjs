// Captures browser console from an already-running dev server.
// Usage: node test-console.mjs [url]

import puppeteer from "puppeteer";

const url = process.argv[2] || "http://localhost:5199";

const browser = await puppeteer.launch({
  headless: "new",
  args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--no-sandbox"],
});

const page = await browser.newPage();

const messages = [];
page.on("console", (msg) => {
  const entry = `[${msg.type()}] ${msg.text()}`;
  messages.push(entry);
  console.log(entry);
});

page.on("pageerror", (err) => {
  const entry = `[PAGE ERROR] ${err.message}`;
  messages.push(entry);
  console.log(entry);
});

try {
  await page.goto(url, { waitUntil: "networkidle0", timeout: 10000 });
} catch (e) {
  console.log(`[navigation] ${e.message}`);
}

await new Promise((r) => setTimeout(r, 3000));

console.log("\n--- Done. Collected", messages.length, "console messages ---");
await browser.close();
process.exit(0);
