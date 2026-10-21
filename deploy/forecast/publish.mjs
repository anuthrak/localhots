// Upload the forecast files to the site's Vercel Blob store (needs BLOB_READ_WRITE_TOKEN).
//   node deploy/forecast/publish.mjs out/
import { readdir, readFile } from "node:fs/promises";
import { join } from "node:path";
import { put } from "@vercel/blob";

const dir = process.argv[2];
for (const name of (await readdir(dir)).filter((f) => f.endsWith(".json"))) {
  const blob = await put(`forecast/lightning/${name}`, await readFile(join(dir, name)), {
    access: "public", addRandomSuffix: false, allowOverwrite: true, contentType: "application/json",
    // latest.json and track.json change every night, so they're cached briefly; dated forecasts never change.
    cacheControlMaxAge: ["latest.json", "track.json"].includes(name) ? 300 : 31536000,
  });
  console.log(`published ${blob.url}`);
}
