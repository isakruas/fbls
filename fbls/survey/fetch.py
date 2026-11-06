"""HTTP download in parallel byte ranges with curl.

The MAST mirror limits throughput per connection, so a file is fetched as several
ranges at once, each appended to its own part file (resumable), then joined and
renamed into place. Small files and servers without a content length go in one piece.
"""
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SEGMENTS = 8
MIN_SEGMENT = 4 << 20
CURL = ["curl", "--fail", "--silent", "--show-error", "--location", "--connect-timeout", "20",
        "--speed-limit", "1000", "--speed-time", "120"]


class NotFound(Exception):
    pass


def size_of(url):
    """Content length after redirects, None when the server does not say; NotFound on 404."""
    head = subprocess.run(CURL + ["--head", url], capture_output=True, text=True)
    if head.returncode == 22 and "404" in head.stderr:
        raise NotFound(url)
    if head.returncode != 0:
        return None
    lengths = [line.split(":", 1)[1].strip() for line in head.stdout.splitlines() if line.lower().startswith("content-length:")]
    return int(lengths[-1]) if lengths and lengths[-1].isdigit() else None


def transfer(url, part, start=None, end=None, attempts=3):
    """Append the bytes of `url` (a range when given) to `part`, resuming from what it already holds."""
    for attempt in range(attempts):
        have = part.stat().st_size if part.exists() else 0
        if start is not None:
            if start + have > end:
                return
            args = ["--range", f"{start + have}-{end}"]
        else:
            args = ["--continue-at", str(have)] if have else []
        with part.open("ab") as out:
            result = subprocess.run(CURL + args + ["--output", "-", url], stdout=out, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            return
        if result.returncode == 22 and "404" in result.stderr:
            raise NotFound(url)
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"transfer failed {url}: {result.stderr.strip()[-200:]}")


def download(url, output, segments=SEGMENTS, min_segment=MIN_SEGMENT):
    """Fetch `url` into `output` (written atomically). False on 404; raises on other failures."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        size = size_of(url)
        if size is None or size < 2 * min_segment:
            part = output.with_name(output.name + ".part")
            transfer(url, part)
            part.replace(output)
            return True
        n = max(1, min(segments, size // min_segment))
        bounds = [(i * size // n, (i + 1) * size // n - 1) for i in range(n)]
        parts = [output.with_name(f"{output.name}.part{i}") for i in range(n)]
        with ThreadPoolExecutor(n) as pool:
            list(pool.map(lambda j: transfer(url, parts[j], *bounds[j]), range(n)))
        joined = output.with_name(output.name + ".part")
        with joined.open("wb") as out:
            for part, (start, end) in zip(parts, bounds):
                data = part.read_bytes()
                if len(data) != end - start + 1:
                    raise RuntimeError(f"segment {part.name}: {len(data)} bytes, expected {end - start + 1}")
                out.write(data)
        joined.replace(output)
        for part in parts:
            part.unlink(missing_ok=True)
        return True
    except NotFound:
        return False
