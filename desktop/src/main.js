"use strict";
/* View Finder desktop app: the World Engine's viewer in a native window,
 * the engine as a child process on stdio (no network port), and the whole
 * thing on the NVIDIA GPU (gpu.js). */

const { app, BrowserWindow, Menu, dialog, ipcMain, protocol, session, shell } = require("electron");
const crypto = require("crypto");
const fs = require("fs");
const path = require("path");

const gpu = require("./gpu");
const { Engine, parseArgs } = require("./engine");
const { findPython, locate } = require("./paths");
const { setupRuntime } = require("./runtime");
const { ID, listWorlds, readWorld } = require("./worlds");

const MAX_PHOTO_BYTES = 25 * 1024 * 1024;
const QUALITIES = new Set(["mobile", "desktop", "high"]);
const PHOTO_EXTENSIONS = ["jpg", "jpeg", "png", "webp", "gif", "bmp", "tif", "tiff", "heic", "heif", "avif"];
const VIEWER_FILES = new Set(["index.html", "viewer.css", "glb.js", "renderer.js", "controls.js", "app.js"]);
const UI_FILES = new Set(["setup.html", "setup.css", "setup.js"]);
const CONTENT_TYPES = { ".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8" };
const CSP = [
  "default-src 'none'",
  "script-src 'self'",
  "style-src 'self'",
  "img-src 'self' blob: data:",
  "connect-src 'self'",
  "base-uri 'none'",
  "form-action 'none'",
].join("; ");
const VIEWER_URL = "app://viewer/";
const SETUP_URL = "app://app/setup.html";
const GPU_RETRY_FLAG = "--vf-gpu-retry";

const gpuPolicy = gpu.policy();
let paths = null;
let engine = null;
let mainWindow = null;
let viewer = { gpu: "unknown", nvidia: false, problems: [] };
let settingUp = false;

/* ---------------- before ready: GPU selection ---------------- */
protocol.registerSchemesAsPrivileged([
  { scheme: "app", privileges: { standard: true, secure: true, supportFetchAPI: true } },
]);
gpu.applySwitches(app, gpuPolicy);
let starting = app.requestSingleInstanceLock();
if (!starting) {
  app.quit();
} else if (gpuPolicy === "nvidia" && process.platform === "linux" && gpu.linuxNeedsOffload()) {
  Object.assign(process.env, gpu.LINUX_OFFLOAD_ENV); // inherited by the relaunched app
  starting = false;
  app.relaunch();
  app.exit(0);
} else if (gpuPolicy === "nvidia" && process.platform === "win32") {
  try {
    gpu.ensureWindowsPreference(process.execPath); // before the GPU process starts
  } catch (error) {
    console.error("Could not set the Windows GPU preference:", error.message);
  }
}

/* ---------------- engine ---------------- */
function startEngine(python) {
  engine = new Engine({
    python,
    engineDir: paths.engineDir,
    store: paths.store,
    logFile: path.join(paths.logs, "engine.log"),
    gpuPolicy,
    env: gpu.engineEnv(gpuPolicy),
    extraArgs: parseArgs(process.env.VIEWFINDER_ENGINE_ARGS),
  });
  engine.start();
}

async function restartEngine() {
  if (engine) await engine.stop();
  const python = findPython(paths.pythonCandidates);
  if (!python) {
    mainWindow.loadURL(SETUP_URL);
    return;
  }
  startEngine(python);
  mainWindow.loadURL(VIEWER_URL);
}

async function engineStatus() {
  const status = engine
    ? await engine.status()
    : { state: "failed", error: "The engine's Python environment is not set up.", gpu: {} };
  return { ...status, viewer };
}

async function submitPhoto(bytes, quality, name) {
  if (!(bytes instanceof Uint8Array) || bytes.length === 0) throw new Error("Empty photo");
  if (bytes.length > MAX_PHOTO_BYTES) throw new Error("Photo is larger than 25 MB");
  if (!QUALITIES.has(quality)) throw new Error("Unknown quality");
  if (!engine) throw new Error("The engine is not running.");
  fs.mkdirSync(paths.uploads, { recursive: true });
  const file = path.join(paths.uploads, `${crypto.randomUUID()}.photo`);
  await fs.promises.writeFile(file, bytes);
  try {
    return await engine.request("build", { path: file, quality, delete: true, name: String(name || "").slice(0, 255) });
  } catch (error) {
    fs.promises.unlink(file).catch(() => {});
    throw error;
  }
}

/* ---------------- IPC: the viewer's window.viewfinderDesktop ---------------- */
function handle(channel, fn) {
  ipcMain.handle(channel, async (event, ...args) => {
    const url = (event.senderFrame && event.senderFrame.url) || "";
    if (!url.startsWith("app://")) return { ok: false, error: "Not allowed" };
    try {
      return { ok: true, result: await fn(...args) };
    } catch (error) {
      return { ok: false, error: error.message };
    }
  });
}

function registerIpc() {
  handle("engine:status", engineStatus);
  handle("engine:build", submitPhoto);
  handle("engine:job", (jobId) => {
    if (!ID.test(String(jobId))) throw new Error("Unknown job");
    if (!engine) throw new Error("The engine is not running.");
    return engine.request("job", { job_id: jobId });
  });
  handle("worlds:list", () => listWorlds(paths.store));
  // The viewer's own WebGL renderer: the surest word on which GPU draws it.
  handle("gpu:report", (renderer) => {
    const reported = gpu.reportedGpu(renderer);
    if (!reported.known) return viewer;
    viewer = gpu.viewerStatus(reported, gpuPolicy);
    if (gpuPolicy === "nvidia" && viewer.problems.length) setImmediate(refuseGpu);
    return viewer;
  });
  handle("worlds:read", (worldId) => readWorld(paths.store, worldId));
  handle("setup:start", async () => {
    if (settingUp) throw new Error("Setup is already running");
    settingUp = true;
    try {
      const log = (text) => mainWindow && mainWindow.webContents.send("setup:log", text);
      await setupRuntime({ paths, log });
    } finally {
      settingUp = false;
    }
    await restartEngine();
  });
}

/* ---------------- app:// pages ---------------- */
function registerProtocol() {
  protocol.handle("app", async (request) => {
    const url = new URL(request.url);
    let file = null;
    if (url.host === "viewer") {
      const name = url.pathname === "/" || url.pathname === "/index.html"
        ? "index.html"
        : url.pathname.startsWith("/static/") ? url.pathname.slice("/static/".length) : null;
      if (name && VIEWER_FILES.has(name)) file = path.join(paths.viewerDir, name);
    } else if (url.host === "app") {
      const name = url.pathname.slice(1);
      if (UI_FILES.has(name)) file = path.join(__dirname, "ui", name);
    }
    if (!file) return new Response("Not found", { status: 404 });
    const body = await fs.promises.readFile(file);
    return new Response(body, {
      headers: {
        "content-type": CONTENT_TYPES[path.extname(file)] || "application/octet-stream",
        "content-security-policy": CSP,
        "x-content-type-options": "nosniff",
      },
    });
  });
}

/* ---------------- mainWindow and menu ---------------- */
async function openPhoto() {
  if (!mainWindow) return;
  const result = await dialog.showOpenDialog(mainWindow, {
    title: "Open a photo",
    properties: ["openFile"],
    filters: [{ name: "Photos", extensions: PHOTO_EXTENSIONS }],
  });
  if (result.canceled || !result.filePaths.length) return;
  const file = result.filePaths[0];
  const { size } = await fs.promises.stat(file);
  if (size > MAX_PHOTO_BYTES) {
    dialog.showMessageBox(mainWindow, { type: "warning", message: "That photo is larger than 25 MB." });
    return;
  }
  mainWindow.webContents.send("menu:open-photo", await fs.promises.readFile(file), path.basename(file));
}

async function showGpuStatus() {
  const status = await engineStatus();
  const engineGpu = status.gpu || {};
  const lines = [
    `Policy: ${gpuPolicy === "nvidia" ? "NVIDIA GPU only" : "any GPU (VIEWFINDER_GPU=any)"}`,
    `3D view: ${viewer.gpu}`,
    `Engine models (CUDA): ${engineGpu.cuda || "none"}`,
    `Engine renderer (OpenGL): ${engineGpu.renderer || "not used"}`,
  ];
  const problems = [...(viewer.problems || []), ...(engineGpu.problems || [])];
  if (problems.length) lines.push("", ...problems);
  dialog.showMessageBox(mainWindow, { type: problems.length ? "warning" : "info", title: "GPU status", message: "GPU status", detail: lines.join("\n") });
}

function buildMenu() {
  const isMac = process.platform === "darwin";
  const template = [
    ...(isMac ? [{ role: "appMenu" }] : []),
    {
      label: "File",
      submenu: [
        { label: "Open Photo…", accelerator: "CmdOrCtrl+O", click: openPhoto },
        { type: "separator" },
        { label: "Show Worlds Folder", click: () => shell.openPath(paths.store) },
        { label: "Show Engine Log", click: () => shell.openPath(path.join(paths.logs, "engine.log")) },
        { label: "Restart Engine", click: restartEngine },
        { type: "separator" },
        isMac ? { role: "close" } : { role: "quit" },
      ],
    },
    {
      label: "View",
      submenu: [
        { role: "togglefullscreen" },
        { type: "separator" },
        { role: "reload" },
        { role: "toggleDevTools" },
      ],
    },
    {
      label: "Help",
      submenu: [
        { label: "GPU Status", click: showGpuStatus },
        { label: "About View Finder", click: () => dialog.showMessageBox(mainWindow, {
          title: "View Finder",
          message: `View Finder ${app.getVersion()}`,
          detail: "Step into a photo. Everything runs on this computer.",
        }) },
      ],
    },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

function createWindow(url) {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 820,
    minWidth: 640,
    minHeight: 480,
    title: "View Finder",
    backgroundColor: "#0e1116",
    show: false,
    icon: path.join(__dirname, "..", "build", "icon.png"),
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      sandbox: true,
      nodeIntegration: false,
      spellcheck: false,
    },
  });
  mainWindow.once("ready-to-show", () => mainWindow.show());
  mainWindow.on("closed", () => {
    mainWindow = null;
  });
  mainWindow.loadURL(url);
}

/* A GPU other than NVIDIA under the nvidia policy: try once more (Windows
 * reads the preference set above at process start), then explain and quit. */
let refusing = false;
function refuseGpu() {
  if (refusing) return;
  refusing = true;
  if (process.platform === "win32" && !process.argv.includes(GPU_RETRY_FLAG)) {
    app.relaunch({ args: [...process.argv.slice(1), GPU_RETRY_FLAG] });
    app.exit(0);
    return;
  }
  const windows = process.platform === "win32";
  const choice = dialog.showMessageBoxSync({
    type: "error",
    title: "View Finder needs the NVIDIA GPU",
    message: "View Finder runs only on the NVIDIA GPU.",
    detail:
      `${viewer.problems.join(" ")}\n\n` +
      (windows
        ? "Open Settings > System > Display > Graphics, find View Finder, and set it to High performance (your NVIDIA GPU). Make sure the NVIDIA driver is installed and the laptop is not in an iGPU-only mode."
        : "Install the NVIDIA driver, then start View Finder again.") +
      "\n\nTo run on another GPU anyway, start the app with VIEWFINDER_GPU=any.",
    buttons: windows ? ["Open Graphics settings", "Quit"] : ["Quit"],
    defaultId: 0,
    cancelId: windows ? 1 : 0,
  });
  if (windows && choice === 0) shell.openExternal(gpu.GRAPHICS_SETTINGS_URL);
  app.quit();
}

/* ---------------- lifecycle ---------------- */
app.on("web-contents-created", (_event, contents) => {
  contents.setWindowOpenHandler(() => ({ action: "deny" }));
  contents.on("will-navigate", (event, url) => {
    if (!url.startsWith("app://")) event.preventDefault();
  });
});

app.on("second-instance", () => {
  if (mainWindow) {
    if (mainWindow.isMinimized()) mainWindow.restore();
    mainWindow.focus();
  }
});

app.whenReady().then(async () => {
  if (!starting) return;
  paths = locate({
    packaged: app.isPackaged,
    resourcesPath: process.resourcesPath,
    appDir: path.join(__dirname, ".."),
    userData: app.getPath("userData"),
  });
  session.defaultSession.setPermissionRequestHandler((_contents, _permission, callback) => callback(false));
  registerProtocol();
  registerIpc();
  buildMenu();

  viewer = gpu.viewerStatus(await app.getGPUInfo("complete"), gpuPolicy);
  if (gpuPolicy === "nvidia" && viewer.problems.length) {
    refuseGpu();
    return;
  }

  const python = findPython(paths.pythonCandidates);
  if (python) startEngine(python);
  createWindow(python ? VIEWER_URL : SETUP_URL);
});

let quitting = false;
app.on("before-quit", (event) => {
  if (quitting || !engine) return;
  event.preventDefault();
  quitting = true;
  engine.stop().finally(() => app.quit());
});

app.on("window-all-closed", () => {
  app.quit();
});
