"use strict";
const test = require("node:test");
const assert = require("node:assert");
const gpu = require("../src/gpu");

const NVIDIA_INFO = {
  gpuDevice: [
    { vendorId: 0x8086, deviceId: 0xa7a0, active: false, deviceString: "Intel(R) Iris(R) Xe Graphics" },
    { vendorId: 0x10de, deviceId: 0x28e0, active: true, deviceString: "NVIDIA GeForce RTX 4060 Laptop GPU" },
  ],
  auxAttributes: { glRenderer: "ANGLE (NVIDIA, NVIDIA GeForce RTX 4060 Laptop GPU Direct3D11 vs_5_0 ps_5_0, D3D11)" },
};
const INTEL_INFO = {
  gpuDevice: [
    { vendorId: 0x8086, deviceId: 0xa7a0, active: true },
    { vendorId: 0x10de, deviceId: 0x28e0, active: false },
  ],
  auxAttributes: { glRenderer: "ANGLE (Intel, Intel(R) Iris(R) Xe Graphics Direct3D11 vs_5_0 ps_5_0, D3D11)" },
};
const SWIFTSHADER_INFO = {
  gpuDevice: [{ vendorId: 0xffff, deviceId: 0xffff, active: true }],
  auxAttributes: { glRenderer: "ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (Subzero) (0x0000C0DE)), SwiftShader driver)" },
};

test("policy is NVIDIA-only unless VIEWFINDER_GPU=any", () => {
  assert.strictEqual(gpu.policy({}), "nvidia");
  assert.strictEqual(gpu.policy({ VIEWFINDER_GPU: "nvidia" }), "nvidia");
  assert.strictEqual(gpu.policy({ VIEWFINDER_GPU: " ANY " }), "any");
  assert.strictEqual(gpu.policy({ VIEWFINDER_GPU: "integrated" }), "nvidia"); // no way back to the iGPU
});

test("the RTX 4060 is recognised as the 3D view's GPU", () => {
  const status = gpu.viewerStatus(NVIDIA_INFO, "nvidia");
  assert.strictEqual(status.nvidia, true);
  assert.strictEqual(status.gpu, "NVIDIA GeForce RTX 4060 Laptop GPU");
  assert.deepStrictEqual(status.problems, []);
});

test("the integrated GPU is a problem under the NVIDIA policy", () => {
  const status = gpu.viewerStatus(INTEL_INFO, "nvidia");
  assert.strictEqual(status.nvidia, false);
  assert.match(status.gpu, /Intel/);
  assert.match(status.problems[0], /not the NVIDIA GPU/);
  assert.deepStrictEqual(gpu.viewerStatus(INTEL_INFO, "any").problems, []);
});

test("software rendering never counts as the NVIDIA GPU", () => {
  const status = gpu.viewerStatus(SWIFTSHADER_INFO, "nvidia");
  assert.strictEqual(status.software, true);
  assert.strictEqual(status.nvidia, false);
  assert.strictEqual(status.gpu, "software rendering (no GPU)");
});

test("Chromium is asked for the high-performance GPU", () => {
  const switches = [];
  const app = { commandLine: { appendSwitch: (name) => switches.push(name) } };
  gpu.applySwitches(app, "nvidia");
  assert.deepStrictEqual(switches, ["force_high_performance_gpu"]);
  gpu.applySwitches(app, "any");
  assert.strictEqual(switches.length, 1);
});

test("Windows: the app is marked High performance once", () => {
  const calls = [];
  let stored = "";
  const run = (command, args) => {
    calls.push(args[0]);
    if (args[0] === "query") {
      if (!stored) throw new Error("not found");
      return `    C:\\View Finder.exe    REG_SZ    ${stored}`;
    }
    stored = args[args.indexOf("/d") + 1];
    return "";
  };
  assert.strictEqual(gpu.ensureWindowsPreference("C:\\View Finder.exe", run), true);
  assert.strictEqual(stored, "GpuPreference=2;");
  assert.strictEqual(gpu.ensureWindowsPreference("C:\\View Finder.exe", run), false);
  assert.deepStrictEqual(calls, ["query", "add", "query"]);
});

test("Windows: other graphics options are kept, a power-saving choice is replaced", () => {
  let stored = "SwapEffectUpgradeEnable=1;GpuPreference=1;";
  const run = (command, args) => {
    if (args[0] === "query") return `\r\nHKEY_CURRENT_USER\\...\r\n    C:\\app.exe    REG_SZ    ${stored}\r\n`;
    stored = args[args.indexOf("/d") + 1];
    return "";
  };
  assert.strictEqual(gpu.ensureWindowsPreference("C:\\app.exe", run), true);
  assert.strictEqual(stored, "SwapEffectUpgradeEnable=1;GpuPreference=2;");
  assert.strictEqual(gpu.ensureWindowsPreference("C:\\app.exe", run), false);
});

test("the engine process is pointed at the NVIDIA GPU on every platform", () => {
  assert.deepStrictEqual(gpu.engineEnv("nvidia", "win32"), {
    CUDA_DEVICE_ORDER: "PCI_BUS_ID", VIEWFINDER_GPU: "nvidia", SHIM_MCCOMPAT: "0x800000001",
  });
  const linux = gpu.engineEnv("nvidia", "linux");
  assert.strictEqual(linux.__NV_PRIME_RENDER_OFFLOAD, "1");
  assert.strictEqual(linux.__GLX_VENDOR_LIBRARY_NAME, "nvidia");
  assert.deepStrictEqual(gpu.engineEnv("any", "win32"), { CUDA_DEVICE_ORDER: "PCI_BUS_ID" });
});

test("Linux relaunches with PRIME offload only where the NVIDIA driver is", () => {
  assert.strictEqual(gpu.linuxNeedsOffload({}, () => true), true);
  assert.strictEqual(gpu.linuxNeedsOffload({ __NV_PRIME_RENDER_OFFLOAD: "1" }, () => true), false);
  assert.strictEqual(gpu.linuxNeedsOffload({}, () => false), false);
});

test("an empty GPU report is unknown, not a reason to refuse", () => {
  const status = gpu.viewerStatus({ gpuDevice: [{ vendorId: 0, deviceId: 0, active: false }] }, "nvidia");
  assert.strictEqual(status.known, false);
  assert.deepStrictEqual(status.problems, []);
  assert.strictEqual(gpu.viewerStatus(undefined, "nvidia").known, false);
});

test("WebGL's own renderer string settles which GPU draws the view", () => {
  const nvidia = gpu.reportedGpu("ANGLE (NVIDIA, NVIDIA GeForce RTX 4060 Laptop GPU (0x000028E0) Direct3D11 vs_5_0 ps_5_0, D3D11)");
  assert.deepStrictEqual([nvidia.known, nvidia.nvidia, nvidia.gpu], [true, true, "NVIDIA GeForce RTX 4060 Laptop GPU"]);
  assert.deepStrictEqual(gpu.viewerStatus(nvidia, "nvidia").problems, []);
  const intel = gpu.viewerStatus(gpu.reportedGpu("ANGLE (Intel, Intel(R) Iris(R) Xe Graphics (0x0000A7A0) Direct3D11 vs_5_0 ps_5_0, D3D11)"), "nvidia");
  assert.strictEqual(intel.gpu, "Intel(R) Iris(R) Xe Graphics");
  assert.match(intel.problems[0], /Intel\(R\) Iris\(R\) Xe Graphics, not the NVIDIA GPU/);
  assert.strictEqual(gpu.reportedGpu("").known, false);
  assert.strictEqual(gpu.reportedGpu("ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (Subzero) (0x0000C0DE)), SwiftShader driver)").nvidia, false);
});
