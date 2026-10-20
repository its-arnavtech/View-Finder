"use strict";
/* Built worlds on disk (the engine's WorldStore): <store>/<id>/world.glb
 * and manifest.json. The app reads them directly; ids are checked so a
 * request can never name a path outside the store. */

const fs = require("fs");
const path = require("path");

const ID = /^[0-9a-f]{32}$/;

function worldFile(store, id, name) {
  if (!ID.test(String(id))) throw new Error("Unknown world");
  return path.join(store, id, name);
}

async function readWorld(store, id) {
  try {
    return await fs.promises.readFile(worldFile(store, id, "world.glb"));
  } catch (error) {
    throw new Error("Unknown world");
  }
}

/* Newest first: [{id, name, profile, created}]. */
async function listWorlds(store) {
  let entries;
  try {
    entries = await fs.promises.readdir(store, { withFileTypes: true });
  } catch (error) {
    return [];
  }
  const worlds = [];
  for (const entry of entries) {
    if (!entry.isDirectory() || !ID.test(entry.name)) continue;
    try {
      const manifestPath = worldFile(store, entry.name, "manifest.json");
      const [manifest, stat] = await Promise.all([
        fs.promises.readFile(manifestPath, "utf8").then(JSON.parse),
        fs.promises.stat(manifestPath),
      ]);
      worlds.push({
        id: entry.name,
        name: manifest.name || "World",
        profile: manifest.profile || null,
        created: stat.mtimeMs,
      });
    } catch (error) {
      // half-written or foreign directory: skip it
    }
  }
  return worlds.sort((a, b) => b.created - a.created);
}

module.exports = { ID, listWorlds, readWorld };
