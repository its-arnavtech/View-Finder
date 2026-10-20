"use strict";
/* The bridge between the sandboxed page and the main process. The viewer
 * (viewfinder/world/viewer/app.js) finds window.viewfinderDesktop and talks
 * to the engine through it instead of over HTTP. */

const { contextBridge, ipcRenderer } = require("electron");

function call(channel, ...args) {
  return ipcRenderer.invoke(channel, ...args).then((reply) => {
    if (reply && reply.ok) return reply.result;
    throw new Error((reply && reply.error) || "The app could not do that.");
  });
}

function toArrayBuffer(bytes) {
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

contextBridge.exposeInMainWorld("viewfinderDesktop", {
  status: () => call("engine:status"),
  build: (bytes, quality, name) => call("engine:build", new Uint8Array(bytes), quality, name),
  job: (jobId) => call("engine:job", jobId),
  world: (worldId) => call("worlds:read", worldId).then(toArrayBuffer),
  worlds: () => call("worlds:list"),
  reportGpu: (renderer) => call("gpu:report", renderer),
  onOpenPhoto: (callback) => {
    ipcRenderer.on("menu:open-photo", (_event, bytes, name) => callback(toArrayBuffer(bytes), name));
  },
});

if (location.host === "app") {
  contextBridge.exposeInMainWorld("viewfinderSetup", {
    start: () => call("setup:start"),
    onLog: (callback) => ipcRenderer.on("setup:log", (_event, text) => callback(text)),
  });
}
