"use strict";
/* The engine child process: `python -m viewfinder.world.cli desktop`,
 * spoken to in JSON lines over stdin/stdout (viewfinder/world/desktop.py).
 * No network port is opened. Its stderr goes to a log file. */

const { spawn } = require("child_process");
const fs = require("fs");
const path = require("path");
const readline = require("readline");

const REQUEST_TIMEOUT_MS = 30000;
const TAIL_LINES = 40;

class Engine {
  constructor({ python, engineDir, store, logFile, gpuPolicy, env = {}, extraArgs = [], spawnFn = spawn }) {
    this.python = python;
    this.engineDir = engineDir;
    this.store = store;
    this.logFile = logFile;
    this.gpuPolicy = gpuPolicy;
    this.env = env;
    this.extraArgs = extraArgs;
    this.spawnFn = spawnFn;
    this.child = null;
    this.nextId = 1;
    this.pending = new Map();
    this.tail = [];
    this.exitError = null;
  }

  start() {
    if (this.child) return;
    this.exitError = null;
    fs.mkdirSync(this.store, { recursive: true });
    if (this.logFile) fs.mkdirSync(path.dirname(this.logFile), { recursive: true });
    const log = this.logFile ? fs.createWriteStream(this.logFile, { flags: "a" }) : null;
    const args = ["-m", "viewfinder.world.cli", "desktop", "--store", this.store, "--gpu", this.gpuPolicy, ...this.extraArgs];
    const child = this.spawnFn(this.python, args, {
      cwd: this.engineDir,
      env: {
        ...process.env,
        ...this.env,
        PYTHONPATH: this.engineDir,
        PYTHONUNBUFFERED: "1",
        PYTHONIOENCODING: "utf-8",
      },
      stdio: ["pipe", "pipe", "pipe"],
      windowsHide: true,
    });
    this.child = child;
    // Writing just as the engine exits fails with EPIPE; the exit handler reports it.
    child.stdin.on("error", () => {});
    readline.createInterface({ input: child.stdout }).on("line", (line) => this._reply(line));
    readline.createInterface({ input: child.stderr }).on("line", (line) => {
      this.tail.push(line);
      if (this.tail.length > TAIL_LINES) this.tail.shift();
      if (log) log.write(line + "\n");
    });
    child.on("error", (error) => this._exited(`The engine could not be started (${error.message}).`));
    child.on("exit", (code, signal) => {
      if (log) log.end();
      this._exited(`The engine stopped unexpectedly (${signal || `exit code ${code}`}). ${this.lastError()}`.trim());
    });
  }

  /* The last meaningful stderr line, for error messages. */
  lastError() {
    const lines = this.tail.filter((line) => /error|exception|failed/i.test(line));
    return lines.length ? lines[lines.length - 1] : "";
  }

  _exited(message) {
    if (!this.child) return;
    this.child = null;
    this.exitError = this.stopping ? null : message;
    for (const { reject, timer } of this.pending.values()) {
      clearTimeout(timer);
      reject(new Error(message));
    }
    this.pending.clear();
  }

  _reply(line) {
    let reply;
    try {
      reply = JSON.parse(line);
    } catch (error) {
      return; // not ours; the engine keeps stdout for replies, but be lenient
    }
    const waiting = this.pending.get(reply.id);
    if (!waiting) return;
    this.pending.delete(reply.id);
    clearTimeout(waiting.timer);
    if (reply.ok) waiting.resolve(reply.result);
    else waiting.reject(new Error(reply.error || "The engine refused the request."));
  }

  request(op, params = {}, timeoutMs = REQUEST_TIMEOUT_MS) {
    if (!this.child) {
      return Promise.reject(new Error(this.exitError || "The engine is not running."));
    }
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`The engine did not answer (${op}).`));
      }, timeoutMs);
      this.pending.set(id, { resolve, reject, timer });
      this.child.stdin.write(JSON.stringify({ id, op, ...params }) + "\n");
    });
  }

  /* Never rejects: a stopped engine is a "failed" status the UI shows. */
  status() {
    if (!this.child) {
      return Promise.resolve({ state: "failed", error: this.exitError || "The engine is not running.", gpu: {} });
    }
    return this.request("status").catch((error) => ({ state: "failed", error: error.message, gpu: {} }));
  }

  stop(graceMs = 5000) {
    const child = this.child;
    if (!child) return Promise.resolve();
    this.stopping = true;
    return new Promise((resolve) => {
      const timer = setTimeout(() => {
        child.kill();
        resolve();
      }, graceMs);
      child.once("exit", () => {
        clearTimeout(timer);
        resolve();
      });
      try {
        child.stdin.write(JSON.stringify({ id: 0, op: "shutdown" }) + "\n");
        child.stdin.end();
      } catch (error) {
        child.kill();
      }
    }).then(() => {
      this.stopping = false;
    });
  }
}

/* "--depth-model heuristic --inpaint none" → ["--depth-model", ...]. */
function parseArgs(text) {
  return String(text || "").split(/\s+/).filter(Boolean);
}

module.exports = { Engine, parseArgs };
