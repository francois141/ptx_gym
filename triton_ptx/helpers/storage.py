from abc import ABC, abstractmethod
import time
from pathlib import Path


class BaseDatasetWriter(ABC):
    """Parent class for storing datasets."""

    def __init__(self, dataset_dir: str = "dataset"):
        self.dataset_dir = Path(dataset_dir)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def store(self, round_index, data: dict):
        """Store dataset and return saved path."""
        pass

    def _generate_filename(self, extra_message) -> str:
        """Generate a nanosecond timestamp filename."""

        if extra_message is not None:
            return f"{time.time_ns()}_{extra_message}.json"

        return f"{time.time_ns()}.json"

class JsonDatasetWriter(BaseDatasetWriter):
    """Stores a dictionary as a JSON file inside the dataset folder."""

    def store(self, round_index, data: dict):
        for idx, res in enumerate(data):
            output_path = self.dataset_dir / f"output_{round_index}_{idx}.json"
            output_path.write_text(
                res.to_json(),
                encoding="utf-8"
            )
