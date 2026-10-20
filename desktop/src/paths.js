"use strict";
/* Where the app finds the engine's code, its Python and its data.
 *
 * From a checkout (`npm start` in desktop/) the engine is the repository's
 * own `viewfinder` package. In an installed app it is a copy under
 * resources/engine. Python is, in order: VIEWFINDER_PYTHON, the runtime
 * the app set up itself (userData/runtime), then the checkout's .venv.
 */

const fs = require("fs");
const path = require("path");

function venvPython(venv, platform = process.platform) {
  return platform === "win32" ? path.join(venv, "Scripts", "python.exe") : path.join(venv, "bin", "python");
}

function locate({ packaged, resourcesPath, appDir, userData, env = process.env, platform = process.platform }) {
  const repo = path.resolve(appDir, "..");
  const engineDir = packaged ? path.join(resourcesPath, "engine") : repo;
  const runtimeVenv = path.join(userData, "runtime", "venv");
  const candidates = [];
  if (env.VIEWFINDER_PYTHON) candidates.push(env.VIEWFINDER_PYTHON);
  candidates.push(venvPython(runtimeVenv, platform));
  if (!packaged) candidates.push(venvPython(path.join(repo, ".venv"), platform), venvPython(path.join(repo, "venv"), platform));
  return {
    engineDir,
    viewerDir: path.join(engineDir, "viewfinder", "world", "viewer"),
    requirements: [
      path.join(engineDir, "viewfinder", "requirements.txt"),
      path.join(engineDir, "viewfinder", "requirements-world.txt"),
    ],
    runtimeVenv,
    pythonCandidates: candidates,
    store: path.join(userData, "worlds"),
    uploads: path.join(userData, "uploads"),
    logs: path.join(userData, "logs"),
  };
}

function findPython(candidates, exists = fs.existsSync) {
  return candidates.find((candidate) => exists(candidate)) || null;
}

module.exports = { findPython, locate, venvPython };
