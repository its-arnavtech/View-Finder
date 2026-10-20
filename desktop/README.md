# View Finder desktop app

The World Engine as a desktop app: open a photo and step into it, in its own
window. It is an [Electron](https://www.electronjs.org/) shell around the same
engine and viewer as `view-finder world serve`, with three differences:

- **No network port.** The engine runs as a child process and the app talks
  to it over stdin/stdout ([`viewfinder/world/desktop.py`](../viewfinder/world/desktop.py)).
  Nothing listens on your network, not even on localhost.
- **NVIDIA GPU only.** Every GPU stage runs on the NVIDIA card (on a laptop,
  the RTX 4060) and never on the integrated GPU. See [GPU](#gpu).
- **Native app.** File > Open Photo (Ctrl+O), drag and drop, recently built
  worlds, a native save dialog for `.glb`, and an installer.

## Run it from this repository

You need Node.js 20 or newer and the repository's Python environment
(`.venv` at the repository root, set up as in the main README, with
`viewfinder/requirements-world.txt` and `view-finder world setup` done).

```powershell
cd desktop
npm install
npm start
```

The app finds `..\.venv\Scripts\python.exe` by itself. To use another
interpreter, set `VIEWFINDER_PYTHON` to its full path.

## Build the installer

```powershell
cd desktop
npm run dist          # Windows: dist\View Finder Setup <version>.exe
npm run dist:linux    # Linux: dist/View Finder-<version>.AppImage
```

The installer holds the app and the engine's Python code, about the size of
any Electron app (~100 MB). It does not hold Python, PyTorch or the models: PyTorch with CUDA and the models
come to about 13 GB. On first launch the app shows **Set up the engine**, which:

1. finds Python 3.11–3.13 on the computer (install it from python.org if
   needed, with "Add python.exe to PATH" ticked);
2. creates a private environment in `%APPDATA%\View Finder\runtime`;
3. installs PyTorch's CUDA build (from `download.pytorch.org/whl/cu130`;
   set `VIEWFINDER_TORCH_INDEX` to use another CUDA version) and the
   engine's requirements;
4. downloads the models (`view-finder world setup`).

A checkout with a `.venv` skips all of this.

## GPU

On a laptop with an NVIDIA GPU and an integrated one, the app keeps
everything on the NVIDIA card:

| What | How it is kept on the NVIDIA GPU |
|---|---|
| The 3D view (WebGL) | On Windows the app marks itself **High performance** in Windows' per-app graphics settings (the same setting as Settings > System > Display > Graphics), and asks Chromium for the high-performance GPU (`force_high_performance_gpu`, WebGL `powerPreference: "high-performance"`). |
| Depth, generation (PyTorch) | CUDA, which only runs on NVIDIA. The engine refuses to start without it. |
| The engine's offscreen renderer (OpenGL) | NVIDIA Optimus (`SHIM_MCCOMPAT`) on Windows, PRIME render offload on Linux. |

It also checks. At start-up the viewer reports the GPU WebGL actually runs
on: if that is not NVIDIA, the app says which GPU it got and quits (on
Windows it first restarts once, because Windows applies the graphics
setting at process start). The engine checks its CUDA device and OpenGL
renderer the same way and refuses to start on anything else; the app then
shows why. **Help > GPU Status** shows what each part runs on.

If it stops anyway, open Settings > System > Display > Graphics, find View
Finder (or `electron.exe` when run from the repository), choose **Options**,
then **High performance**. Also check that the laptop is not in an
integrated-only (Eco/Optimus off) mode in its vendor tool.

`VIEWFINDER_GPU=any` lifts the requirement, for a computer without an
NVIDIA GPU.

## Settings (environment variables)

| Variable | Effect |
|---|---|
| `VIEWFINDER_GPU` | `nvidia` (default): NVIDIA GPU only. `any`: whatever the system picks. |
| `VIEWFINDER_PYTHON` | Full path to the Python that runs the engine. |
| `VIEWFINDER_ENGINE_ARGS` | Extra `view-finder world desktop` options, e.g. `--depth-model heuristic --generate off`. |
| `VIEWFINDER_TORCH_INDEX` | PyTorch wheel index for first-run setup (default CUDA 13.0). |

## Where things are

| | Windows | Linux |
|---|---|---|
| Worlds (newest 30) | `%APPDATA%\View Finder\worlds` | `~/.config/View Finder/worlds` |
| Engine log | `%APPDATA%\View Finder\logs\engine.log` | `~/.config/View Finder/logs/engine.log` |
| First-run environment | `%APPDATA%\View Finder\runtime` | `~/.config/View Finder/runtime` |

File > Show Worlds Folder and File > Show Engine Log open them.

## How it fits together

```
Electron main process (src/main.js)
  ├─ gpu.js       NVIDIA-only policy: Windows GPU preference, Chromium switch, checks
  ├─ engine.js    spawns: python -m viewfinder.world.cli desktop --store … --gpu nvidia
  │               JSON lines on stdin/stdout: status, build, job
  ├─ worlds.js    reads built worlds straight from the store
  ├─ runtime.js   first-run Python environment
  └─ app://viewer/  the engine's own viewer (viewfinder/world/viewer), served
                  from disk with a strict Content-Security-Policy
Renderer (sandboxed, no Node)
  └─ viewer app.js finds window.viewfinderDesktop (src/preload.js) and uses
     it instead of HTTP
```

## Tests

```bash
cd desktop && npm test                                   # the app's own modules
python -m pytest viewfinder/tests/test_world_desktop.py  # the engine side
```
