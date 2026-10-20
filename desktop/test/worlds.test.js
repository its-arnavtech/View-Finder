"use strict";
const test = require("node:test");
const assert = require("node:assert");
const fs = require("fs");
const os = require("os");
const path = require("path");
const { listWorlds, readWorld } = require("../src/worlds");

function store() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "vf-worlds-"));
  const add = (id, name, when) => {
    fs.mkdirSync(path.join(root, id));
    fs.writeFileSync(path.join(root, id, "world.glb"), Buffer.from(`glb ${name}`));
    fs.writeFileSync(path.join(root, id, "manifest.json"), JSON.stringify({ name, profile: "desktop" }));
    fs.utimesSync(path.join(root, id, "manifest.json"), when, when);
  };
  add("a".repeat(32), "lake", new Date(2026, 0, 1));
  add("b".repeat(32), "forest", new Date(2026, 5, 1));
  fs.mkdirSync(path.join(root, ".cccc.tmp")); // a world being written
  fs.mkdirSync(path.join(root, "c".repeat(32))); // no manifest
  return root;
}

test("worlds are listed newest first, skipping incomplete ones", async () => {
  const worlds = await listWorlds(store());
  assert.deepStrictEqual(worlds.map((w) => w.name), ["forest", "lake"]);
  assert.strictEqual(worlds[0].profile, "desktop");
});

test("a world is read by id, and only by id", async () => {
  const root = store();
  assert.strictEqual((await readWorld(root, "a".repeat(32))).toString(), "glb lake");
  await assert.rejects(readWorld(root, "../../etc/passwd"), /Unknown world/);
  await assert.rejects(readWorld(root, "d".repeat(32)), /Unknown world/);
});

test("a missing store is an empty list", async () => {
  assert.deepStrictEqual(await listWorlds(path.join(os.tmpdir(), "vf-missing-store")), []);
});
