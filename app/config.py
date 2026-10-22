import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    storage: Path
    dataset: Path
    max_upload_bytes: int = 15 * 1024 * 1024
    max_pixels: int = 25_000_000

    @classmethod
    def from_env(cls):
        return cls(
            Path(os.getenv("BONE_STORAGE", str(ROOT / "var"))).resolve(),
            Path(os.getenv("BONE_DATASET", "/workspace/bone-data" if Path("/workspace/bone-data").is_dir()
                           else str(ROOT / "data"))).resolve(),
        )

    def prepare(self):
        for name in ("images", "explanations", "models"):
            (self.storage / name).mkdir(parents=True, exist_ok=True)

    @property
    def database(self):
        return self.storage / "workbench.sqlite3"
