from pathlib import Path 
import shutil

def clear_triton_cache() -> None:
    cache = Path.home() / ".triton" / "cache"
    if cache.exists():
        shutil.rmtree(cache)
        print(f"Cache cleared: {cache}")