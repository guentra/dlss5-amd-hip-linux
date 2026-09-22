"""GPU targets supported by the bundled native HIP network library."""
from __future__ import annotations

from pathlib import Path
import re

FALLBACK_TARGETS = frozenset(('gfx1200', 'gfx1201'))
# Device images are identified by the amdgcn triple of the fat container child
# (e.g. "hipv4-amdgcn-amd-amdhsa--gfx1201"). A bare "gfxNNNN" token scan also
# matches compiler metadata names (such as "gfx1250_rev") that carry no image.
_TARGET_RE = re.compile(rb'amdgcn-amd-amdhsa--gfx[0-9a-f]{3,5}')
_TARGET_PREFIX = b'amdgcn-amd-amdhsa--'


def bundled_targets(so_path) -> frozenset:
    """gfx targets embedded in the bundled libdlss5_hip.so kernel images."""
    try:
        data = Path(so_path).read_bytes()
    except OSError as e:
        raise RuntimeError(f'Cannot read bundled HIP library {so_path}: {e}') from e
    found = frozenset(target.decode('ascii')[len(_TARGET_PREFIX):]
                      for target in _TARGET_RE.findall(data))
    return found or FALLBACK_TARGETS
