"use strict";
const test = require("node:test");
const assert = require("node:assert");
const fs = require("fs");
const os = require("os");
const path = require("path");
const runtime = require("../src/runtime");

test("the CUDA torch pins come from requirements.txt", () => {
  const text = fs.readFileSync(path.join(__dirname, "..", "..", "viewfinder", "requirements.txt"), "utf8");
  const pins = runtime.torchPins(text);
  assert.deepStrictEqual(pins.map((p) => p.split("==")[0]), ["torch", "torchvision"]);
});

test("Python 3.11-3.13 is found through the Windows launcher", async () => {
  const tried = [];
  const runFn = async (command, args) => {
    tried.push(`${command} ${args[0]}`);
    if (args[0] === "-3.12") return "3.12\n";
    throw new Error("not installed");
  };
  const found = await runtime.findBasePython({ platform: "win32", runFn });
  assert.deepStrictEqual(found, { command: "py", prefix: ["-3.12"], version: "3.12" });
  assert.deepStrictEqual(tried, ["py -3.13", "py -3.12"]);
});

test("an unsupported Python is skipped", async () => {
  const runFn = async (command) => (command === "python3" ? "3.9\n" : Promise.reject(new Error("missing")));
  assert.strictEqual(await runtime.findBasePython({ platform: "linux", runFn }), null);
});

test("setup on Windows: venv, CUDA torch, requirements, then models", async () => {
  const userData = fs.mkdtempSync(path.join(os.tmpdir(), "vf-runtime-"));
  const engineDir = path.join(__dirname, "..", "..");
  const paths = {
    engineDir,
    runtimeVenv: path.join(userData, "runtime", "venv"),
    requirements: [
      path.join(engineDir, "viewfinder", "requirements.txt"),
      path.join(engineDir, "viewfinder", "requirements-world.txt"),
    ],
  };
  const calls = [];
  const runFn = async (command, args, options = {}) => {
    calls.push({ command, args, options });
    return args.includes("-c") ? "3.13\n" : "";
  };
  const logged = [];
  await runtime.setupRuntime({ paths, log: (t) => logged.push(t), platform: "win32", env: {}, runFn });
  const venvPython = path.join(paths.runtimeVenv, "Scripts", "python.exe");
  assert.strictEqual(calls[1].args.join(" "), `-3.13 -m venv ${paths.runtimeVenv}`);
  assert.ok(calls.slice(2).every((c) => c.command === venvPython));
  assert.match(calls[3].args.join(" "), /^-m pip install --index-url https:\/\/download\.pytorch\.org\/whl\/cu130 /);
  assert.ok(calls[3].args.some((a) => a.startsWith("torch==")));
  assert.deepStrictEqual(calls[4].args.filter((a) => a === "-r").length, 2);
  assert.deepStrictEqual(calls[5].args, ["-m", "viewfinder.world.cli", "setup"]);
  assert.strictEqual(calls[5].options.env.PYTHONPATH, engineDir);
  assert.match(logged.join(""), /The engine is set up/);
});

test("setup explains a missing Python", async () => {
  const runFn = async () => { throw new Error("missing"); };
  await assert.rejects(
    runtime.setupRuntime({ paths: {}, log: () => {}, platform: "win32", env: {}, runFn }),
    /Python 3\.11, 3\.12 or 3\.13 is needed/
  );
});
