"use strict";
/* First launch of an installed app: create the engine's Python environment.
 *
 *   1. find Python 3.11-3.13 on this computer;
 *   2. create a venv under userData/runtime;
 *   3. install PyTorch's CUDA build (Windows; PyPI's Windows wheels are
 *      CPU-only, and the engine runs on the NVIDIA GPU);
 *   4. install the engine's requirements;
 *   5. download the models (`view-finder world setup`, ~9 GB).
 *
 * Each step streams its output to `log`. Nothing here runs on a checkout
 * that already has a .venv (paths.js finds that first).
 */

const { spawn } = require("child_process");
const fs = require("fs");
const path = require("path");
const { venvPython } = require("./paths");

const SUPPORTED = [[3, 13], [3, 12], [3, 11]];
// CUDA build of the torch pinned in viewfinder/requirements.txt.
const DEFAULT_TORCH_INDEX = "https://download.pytorch.org/whl/cu130";

function run(command, args, { log, env, cwd } = {}) {
  return new Promise((resolve, reject) => {
    let child;
    try {
      child = spawn(command, args, { env: { ...process.env, ...env }, cwd, windowsHide: true });
    } catch (error) {
      reject(error);
      return;
    }
    let output = "";
    const collect = (chunk) => {
      const text = chunk.toString();
      output += text;
      if (log) log(text);
    };
    child.stdout.on("data", collect);
    child.stderr.on("data", collect);
    child.on("error", reject);
    child.on("exit", (code) => (code === 0 ? resolve(output) : reject(new Error(`${path.basename(command)} exited with ${code}`))));
  });
}

/* Python launchers to try, best first. */
function pythonCandidates(platform = process.platform) {
  if (platform === "win32") {
    return [...SUPPORTED.map(([a, b]) => ["py", [`-${a}.${b}`]]), ["python", []]];
  }
  return [...SUPPORTED.map(([a, b]) => [`python${a}.${b}`, []]), ["python3", []]];
}

async function findBasePython({ platform = process.platform, runFn = run } = {}) {
  for (const [command, prefix] of pythonCandidates(platform)) {
    try {
      const out = await runFn(command, [...prefix, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"]);
      const version = out.trim().split(/\s+/).pop();
      if (SUPPORTED.some(([a, b]) => version === `${a}.${b}`)) return { command, prefix, version };
    } catch (error) {
      // not installed: try the next one
    }
  }
  return null;
}

/* The pinned torch/torchvision lines from requirements.txt. */
function torchPins(requirementsText) {
  return requirementsText
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => /^(torch|torchvision)==/.test(line));
}

async function setupRuntime({ paths, log, platform = process.platform, env = process.env, runFn = run }) {
  const step = (text) => log(`\n== ${text}\n`);
  step("Looking for Python 3.11-3.13");
  const base = await findBasePython({ platform, runFn });
  if (!base) {
    throw new Error(
      "Python 3.11, 3.12 or 3.13 is needed once to set up the engine. Install it from python.org " +
        "(tick \"Add python.exe to PATH\"), then click Set up again."
    );
  }
  log(`Found Python ${base.version}\n`);

  step("Creating the engine's Python environment");
  fs.mkdirSync(path.dirname(paths.runtimeVenv), { recursive: true });
  await runFn(base.command, [...base.prefix, "-m", "venv", paths.runtimeVenv], { log });
  const python = venvPython(paths.runtimeVenv, platform);
  await runFn(python, ["-m", "pip", "install", "--upgrade", "pip"], { log });

  if (platform === "win32") {
    step("Installing PyTorch for NVIDIA GPUs (CUDA, ~3 GB)");
    const pins = torchPins(fs.readFileSync(paths.requirements[0], "utf8"));
    const index = env.VIEWFINDER_TORCH_INDEX || DEFAULT_TORCH_INDEX;
    await runFn(python, ["-m", "pip", "install", "--index-url", index, ...pins], { log });
  }

  step("Installing the engine");
  const requirementArgs = paths.requirements.flatMap((file) => ["-r", file]);
  await runFn(python, ["-m", "pip", "install", ...requirementArgs], { log });

  step("Downloading the models (~9 GB, once)");
  await runFn(python, ["-m", "viewfinder.world.cli", "setup"], {
    log,
    cwd: paths.engineDir,
    env: { PYTHONPATH: paths.engineDir, PYTHONUNBUFFERED: "1" },
  });
  log("\nThe engine is set up.\n");
  return python;
}

module.exports = { DEFAULT_TORCH_INDEX, findBasePython, pythonCandidates, run, setupRuntime, torchPins };
