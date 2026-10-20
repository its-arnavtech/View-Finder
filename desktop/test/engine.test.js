"use strict";
const test = require("node:test");
const assert = require("node:assert");
const { spawn } = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");
const { Engine, parseArgs } = require("../src/engine");

const FAKE = path.join(__dirname, "fixtures", "fake-engine.js");

function fakeEngine(options = {}) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "vf-engine-"));
  const engine = new Engine({
    python: "python-is-not-used",
    engineDir: dir,
    store: path.join(dir, "worlds"),
    logFile: path.join(dir, "logs", "engine.log"),
    gpuPolicy: "nvidia",
    extraArgs: ["--depth-model", "heuristic"],
    // Runs the fake engine with node, passing the engine's arguments through.
    spawnFn: (_python, args, spawnOptions) => spawn(process.execPath, [FAKE, ...args], spawnOptions),
    ...options,
  });
  engine.start();
  return { engine, dir };
}

test("requests and replies are matched over stdio", async () => {
  const { engine, dir } = fakeEngine();
  const [status, again] = await Promise.all([engine.status(), engine.request("status")]);
  assert.strictEqual(status.state, "ready");
  assert.strictEqual(again.gpu.cuda, "Fake GPU");
  assert.deepStrictEqual(status.args.slice(0, 7), ["-m", "viewfinder.world.cli", "desktop", "--store", path.join(dir, "worlds"), "--gpu", "nvidia"]);
  assert.deepStrictEqual(status.args.slice(7), ["--depth-model", "heuristic"]);
  await assert.rejects(engine.request("explode"), /Unknown operation/);
  await engine.stop();
  assert.match(fs.readFileSync(path.join(dir, "logs", "engine.log"), "utf8"), /fake engine starting/);
});

test("a crash fails pending requests and shows up in status", async () => {
  const { engine } = fakeEngine();
  await assert.rejects(engine.request("crash"), /stopped unexpectedly \(exit code 3\)\. ERROR the engine fell over/);
  const status = await engine.status();
  assert.strictEqual(status.state, "failed");
  assert.match(status.error, /fell over/);
  await assert.rejects(engine.request("status"), /stopped unexpectedly/);
});

test("an engine that does not answer times out", async () => {
  const { engine } = fakeEngine();
  await assert.rejects(engine.request("slow", {}, 200), /did not answer \(slow\)/);
  await engine.stop();
});

test("a missing interpreter is reported, not thrown", async () => {
  const { engine } = fakeEngine({ spawnFn: (...args) => spawn("/nonexistent/python", [], args[2]) });
  const status = await new Promise((resolve) => setTimeout(() => resolve(engine.status()), 200));
  assert.strictEqual(status.state, "failed");
  assert.match(status.error, /could not be started/);
});

test("stopping is clean and quiet", async () => {
  const { engine } = fakeEngine();
  await engine.status();
  await engine.stop();
  assert.strictEqual(engine.exitError, null);
});

test("engine arguments from the environment", () => {
  assert.deepStrictEqual(parseArgs(" --depth-model heuristic  --inpaint none "), ["--depth-model", "heuristic", "--inpaint", "none"]);
  assert.deepStrictEqual(parseArgs(undefined), []);
});
