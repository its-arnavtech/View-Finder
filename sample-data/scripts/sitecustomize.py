"""Force torch device to CPU to avoid MPS segfault in open_clip on this Mac."""
import viewfinder.geolocator as _g
import viewfinder.retrieval_geolocator as _r
import viewfinder.local_landmarks as _lm
_g.resolve_device = lambda *a, **k: "cpu"
_r.resolve_device = lambda *a, **k: "cpu"
_lm.resolve_device = lambda *a, **k: "cpu"
