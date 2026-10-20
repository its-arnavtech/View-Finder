"use strict";
/* GPU policy: the whole app runs on the NVIDIA GPU, never the integrated one.
 *
 * Three things draw or compute, and each is pointed at the NVIDIA card:
 *   - this app's 3D view (Chromium/WebGL): Windows' per-app GPU preference
 *     plus Chromium's force_high_performance_gpu switch, then checked after
 *     start-up and refused if it still landed elsewhere;
 *   - the engine's models (PyTorch): CUDA only exists on NVIDIA;
 *   - the engine's offscreen renderer (OpenGL): NVIDIA Optimus via
 *     SHIM_MCCOMPAT on Windows, PRIME render offload on Linux.
 *
 * VIEWFINDER_GPU=any lifts the requirement (machines without NVIDIA).
 */

const fs = require("fs");
const { execFileSync } = require("child_process");

const NVIDIA_VENDOR_ID = 0x10de;
const WINDOWS_PREFERENCES_KEY = "HKCU\\Software\\Microsoft\\DirectX\\UserGpuPreferences";
// What Settings > System > Display > Graphics writes for "High performance".
const HIGH_PERFORMANCE = "GpuPreference=2;";
const GRAPHICS_SETTINGS_URL = "ms-settings:display-advancedgraphics";

function policy(env = process.env) {
  return String(env.VIEWFINDER_GPU || "").trim().toLowerCase() === "any" ? "any" : "nvidia";
}

/* Chromium switches; before the app is ready. */
function applySwitches(app, gpuPolicy) {
  if (gpuPolicy !== "nvidia") return;
  app.commandLine.appendSwitch("force_high_performance_gpu");
}

/* Windows: mark this executable "High performance" (the NVIDIA GPU), as the
 * Graphics settings page would, keeping any other options stored with it
 * ("SwapEffectUpgradeEnable=1;" and the like). Returns true when it changed. */
function ensureWindowsPreference(exePath, run = execFileSync) {
  let current = "";
  try {
    const output = run("reg", ["query", WINDOWS_PREFERENCES_KEY, "/v", exePath], { encoding: "utf8", windowsHide: true });
    const match = /REG_SZ\s+(.*)$/m.exec(output);
    current = match ? match[1].trim() : "";
  } catch (error) {
    current = ""; // no value yet
  }
  if (current.split(";").includes(HIGH_PERFORMANCE.slice(0, -1))) return false;
  const others = current.split(";").filter((part) => part && !part.startsWith("GpuPreference="));
  const data = [...others, HIGH_PERFORMANCE.slice(0, -1)].join(";") + ";";
  run("reg", ["add", WINDOWS_PREFERENCES_KEY, "/v", exePath, "/t", "REG_SZ", "/d", data, "/f"], {
    windowsHide: true,
  });
  return true;
}

/* Linux with the NVIDIA driver: PRIME render offload has to be in the
 * environment before the GPU process starts, so the app relaunches itself
 * with it once. */
function linuxNeedsOffload(env = process.env, exists = fs.existsSync) {
  return exists("/proc/driver/nvidia/version") && env.__NV_PRIME_RENDER_OFFLOAD !== "1";
}

const LINUX_OFFLOAD_ENV = { __NV_PRIME_RENDER_OFFLOAD: "1", __GLX_VENDOR_LIBRARY_NAME: "nvidia" };

/* Environment for the engine process. */
function engineEnv(gpuPolicy, platform = process.platform) {
  const env = { CUDA_DEVICE_ORDER: "PCI_BUS_ID" };
  if (gpuPolicy !== "nvidia") return env;
  env.VIEWFINDER_GPU = "nvidia";
  if (platform === "win32") env.SHIM_MCCOMPAT = "0x800000001"; // NVIDIA Optimus: discrete
  if (platform === "linux") Object.assign(env, LINUX_OFFLOAD_ENV);
  return env;
}

/* Which GPU the 3D view runs on, from app.getGPUInfo("complete"). */
function describeViewerGpu(info) {
  const devices = (info && info.gpuDevice) || [];
  const active = devices.find((d) => d.active) || devices[0] || null;
  const renderer = (info && info.auxAttributes && info.auxAttributes.glRenderer) || "";
  // "ANGLE (NVIDIA, NVIDIA GeForce RTX 4060 Laptop GPU Direct3D11 vs_5_0 ps_5_0, D3D11)"
  const angle = /^ANGLE \(([^,]+), (.+?)(?: Direct3D| \(0x|, )/.exec(renderer);
  const name = (active && active.deviceString) || (angle ? angle[2] : renderer) || "unknown GPU";
  const software = /swiftshader|llvmpipe|software|basic render/i.test(`${renderer} ${name}`);
  const nvidia = !software && ((active && active.vendorId === NVIDIA_VENDOR_ID) || /nvidia/i.test(renderer));
  // Chromium sometimes reports no device at all (vendor 0, no renderer); the
  // viewer's own WebGL report (reportedGpu) settles it then.
  const known = software || !!renderer || !!(active && active.vendorId);
  return { gpu: software ? "software rendering (no GPU)" : name, nvidia: !!nvidia, software, known };
}

/* The renderer string WebGL itself reports (WEBGL_debug_renderer_info),
 * e.g. "ANGLE (NVIDIA, NVIDIA GeForce RTX 4060 Laptop GPU (0x000028E0) Direct3D11 vs_5_0 ps_5_0, D3D11)". */
function reportedGpu(renderer) {
  return describeViewerGpu({ gpuDevice: [], auxAttributes: { glRenderer: String(renderer || "").slice(0, 300) } });
}

/* The viewer-side status the app shows, with problems under the policy.
 * Takes app.getGPUInfo()'s answer, or one already described (reportedGpu). */
function viewerStatus(info, gpuPolicy) {
  const viewer = info && "known" in info ? { ...info } : describeViewerGpu(info);
  viewer.problems = [];
  if (gpuPolicy === "nvidia" && viewer.known && !viewer.nvidia) {
    viewer.problems.push(`The 3D view is running on ${viewer.gpu}, not the NVIDIA GPU.`);
  }
  return viewer;
}

module.exports = {
  GRAPHICS_SETTINGS_URL,
  HIGH_PERFORMANCE,
  LINUX_OFFLOAD_ENV,
  NVIDIA_VENDOR_ID,
  WINDOWS_PREFERENCES_KEY,
  applySwitches,
  describeViewerGpu,
  engineEnv,
  ensureWindowsPreference,
  linuxNeedsOffload,
  policy,
  reportedGpu,
  viewerStatus,
};
