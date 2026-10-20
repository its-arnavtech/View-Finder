"use strict";
const test = require("node:test");
const assert = require("node:assert");
const path = require("path");
const { findPython, locate } = require("../src/paths");

test("a checkout uses the repository's engine and .venv", () => {
  const p = locate({ packaged: false, resourcesPath: "/r", appDir: "/repo/desktop", userData: "/data", env: {}, platform: "linux" });
  assert.strictEqual(p.engineDir, path.resolve("/repo"));
  assert.strictEqual(p.viewerDir, path.join(path.resolve("/repo"), "viewfinder", "world", "viewer"));
  assert.deepStrictEqual(p.pythonCandidates, [
    path.join("/data", "runtime", "venv", "bin", "python"),
    path.join(path.resolve("/repo"), ".venv", "bin", "python"),
    path.join(path.resolve("/repo"), "venv", "bin", "python"),
  ]);
  assert.strictEqual(p.store, path.join("/data", "worlds"));
});

test("an installed app uses its bundled engine and its own runtime", () => {
  const p = locate({ packaged: true, resourcesPath: "C:\\App\\resources", appDir: "x", userData: "C:\\Data", env: {}, platform: "win32" });
  assert.strictEqual(p.engineDir, path.join("C:\\App\\resources", "engine"));
  assert.deepStrictEqual(p.pythonCandidates, [path.join("C:\\Data", "runtime", "venv", "Scripts", "python.exe")]);
});

test("VIEWFINDER_PYTHON comes first", () => {
  const p = locate({ packaged: true, resourcesPath: "/r", appDir: "x", userData: "/d", env: { VIEWFINDER_PYTHON: "/opt/py" }, platform: "linux" });
  assert.strictEqual(p.pythonCandidates[0], "/opt/py");
  assert.strictEqual(findPython(p.pythonCandidates, (c) => c !== "/opt/py"), p.pythonCandidates[1]);
  assert.strictEqual(findPython(p.pythonCandidates, () => false), null);
});
