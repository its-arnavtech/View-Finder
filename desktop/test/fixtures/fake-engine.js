"use strict";
/* Stands in for `python -m viewfinder.world.cli desktop` in engine tests. */
const readline = require("readline");
process.stderr.write("INFO fake engine starting\n");
readline.createInterface({ input: process.stdin }).on("line", (line) => {
  const request = JSON.parse(line);
  const reply = (body) => process.stdout.write(JSON.stringify({ id: request.id, ...body }) + "\n");
  if (request.op === "status") reply({ ok: true, result: { state: "ready", gpu: { cuda: "Fake GPU" }, args: process.argv.slice(2) } });
  else if (request.op === "shutdown") { reply({ ok: true, result: null }); process.exit(0); }
  else if (request.op === "crash") { process.stderr.write("ERROR the engine fell over\n"); process.exit(3); }
  else if (request.op === "slow") { /* never answers */ }
  else reply({ ok: false, error: `Unknown operation '${request.op}'` });
});
