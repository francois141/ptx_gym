import errno
import shutil
import time
import uuid
from pathlib import Path

def clear_triton_cache() -> None:
    cache = Path.home() / ".triton" / "cache"
    if not cache.exists():
        return

    delete_target = cache
    renamed_cache = cache.with_name(f"{cache.name}.deleting.{uuid.uuid4().hex}")
    try:
        cache.replace(renamed_cache)
        delete_target = renamed_cache
    except FileNotFoundError:
        return
    except OSError:
        # If the rename races with another process, fall back to deleting the
        # live cache path directly.
        delete_target = cache

    for attempt in range(3):
        try:
            shutil.rmtree(delete_target)
            print(f"Cache cleared: {cache}")
            return
        except FileNotFoundError:
            return
        except OSError as exc:
            if exc.errno not in (errno.ENOTEMPTY, errno.EBUSY, errno.EPERM) or attempt == 2:
                print(f"Warning: failed to fully clear Triton cache at {cache}: {exc}")
                return
            time.sleep(0.1 * (attempt + 1))
