# View Finder

View Finder is a local Python CLI that estimates where a travel or landscape
photo or video was captured. It uses GeoCLIP to compare scenery cues such as
architecture, signage, vegetation, and terrain with a geographic gallery. It
does not identify people.

Inference is offline. The CLI forces Hugging Face/Transformers offline mode,
and place names come from `reverse_geocoder`'s bundled dataset. GeoCLIP's model
weights must therefore be downloaded and cached once during setup.

## Requirements

- Python 3.10 or newer
- Enough memory to run GeoCLIP's CLIP ViT-L/14 backbone
- A CUDA GPU is used automatically when available; CPU inference also works
- A local video codec supported by OpenCV for video inputs

## Installation

From the project root:

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r viewfinder/requirements.txt
```

Cache the GeoCLIP backbone once while network access is available:

```powershell
python -c "from viewfinder.geolocator import GeoLocator; GeoLocator()"
```

After that setup step, normal CLI runs do not make network calls. If the model
is not cached, the CLI exits with an actionable error instead of downloading it
silently.

## Usage

Analyze an image:

```powershell
python -m viewfinder.main --input .\photos\coast.jpg
```

Analyze a video at two sampled frames per second and retain ten candidates:

```powershell
python -m viewfinder.main --input .\clips\walk.mov --fps 2 --top-k 10
```

Choose an output path and enable stage-level debug logging:

```powershell
python -m viewfinder.main --input .\photos\street.png --output .\results\street.json -v
```

Produce only the final prediction on stdout (as one JSON line):

```powershell
python -m viewfinder.main --input .\clips\panorama.mp4 --quiet
```

CLI options:

```text
--input PATH       Required image or video path
--fps FLOAT        Video sampling rate; default 1.0
--top-k INT        Ranked candidates retained/reported; default 5
--max-predictions INT  Distinct location clusters returned; default 5
--output PATH      JSON path; defaults beside the input file
--verbose, -v      Debug logging for pipeline stages
--quiet, -q        No progress logging; print only the final prediction
```

Supported images are `.jpg`, `.jpeg`, and `.png`. Supported videos are `.mp4`,
`.mov`, `.avi`, and `.mkv`. Format detection is extension-based and
case-insensitive.

## Pipeline

For an image, View Finder loads and orients the image, retrieves ranked GeoCLIP
coordinates, geographically clusters the top-k results, independently scores
each distinct location hypothesis, and reverse-geocodes every returned cluster.

For a video, it samples at the requested rate, rejects blurry and near-duplicate
frames, runs GeoCLIP in batches, and applies connected-component clustering with
haversine distance to the complete frame x top-k candidate pool. Every cluster
gets its own score-weighted spherical centroid, confidence breakdown, and count
of supporting frames. Clusters remain separate, are ranked by confidence, and
are reverse-geocoded in one offline batch. Progress is reported after inference
batches unless quiet mode is enabled.

## Output

Unless `--output` is supplied, `photo.jpg` produces
`photo_viewfinder.json` in the same directory. A video result resembles:

```json
{
  "input": "C:\\photos\\trip.mp4",
  "input_type": "video",
  "prediction": {
    "latitude": 41.9028,
    "longitude": 12.4964,
    "country": "IT",
    "admin1": "Latium",
    "city": "Rome",
    "confidence_percent": 78.4
  },
  "predictions": [
    {
      "rank": 1,
      "lat": 41.9028,
      "lon": 12.4964,
      "confidence": 78.4,
      "support_count": 6,
      "candidate_count": 12,
      "country": "IT",
      "admin1": "Latium",
      "city": "Rome",
      "confidence_details": {}
    },
    {
      "rank": 2,
      "lat": 43.7696,
      "lon": 11.2558,
      "confidence": 45.2,
      "support_count": 2,
      "candidate_count": 3,
      "country": "IT",
      "admin1": "Tuscany",
      "city": "Florence",
      "confidence_details": {}
    }
  ],
  "alternatives": [
    {
      "rank": 2,
      "lat": 43.7696,
      "lon": 11.2558,
      "confidence": 45.2,
      "support_count": 2,
      "candidate_count": 3,
      "country": "IT",
      "admin1": "Tuscany",
      "city": "Florence",
      "confidence_details": {}
    }
  ],
  "processing": {
    "video_fps": 30.0,
    "duration_seconds": 12.0,
    "total_frames": 360,
    "frames_sampled": 12,
    "frames_kept": 8,
    "frames_skipped_blurry": 2,
    "frames_skipped_duplicate": 2,
    "blur_fallback_used": false,
    "frames_analyzed": 8,
    "frame_dispersion_km": 14.2,
    "confidence_details": {}
  }
}
```

`prediction` remains the highest-confidence result for compatibility with
single-answer consumers. `predictions` contains every ranked cluster up to
`--max-predictions`, and `alternatives` contains the same objects from rank 2
onward. `support_count` is the number of independent source frames backing the
cluster; `candidate_count` is the number of retained top-k entries it contains.
For a still image, support is necessarily one frame. View Finder confidence is
a relative ranking/reliability indicator, not a calibrated probability or a
guarantee that the true location lies within a particular distance.

## Errors and exit codes

- Exit `0`: result written successfully
- Exit `1`: a pipeline stage failed (decoding, model loading/inference,
  aggregation, reverse geocoding, or output writing)
- Exit `2`: invalid CLI input, such as a missing file or unsupported extension

Failures are reported as concise messages without raw tracebacks. Common fixes
are included in the message where possible.
