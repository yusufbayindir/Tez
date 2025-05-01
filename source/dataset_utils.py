# src/dataset_utils.py
from pathlib import Path
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_dataset(root_dir: str) -> pd.DataFrame:
    rows = []
    root = Path(root_dir)
    for source in ["A","B"]:
        src_path = root/source
        if not src_path.exists(): continue
        for class_dir in src_path.iterdir():
            if not class_dir.is_dir() or "_" not in class_dir.name:
                continue
            # class_dir.name örn. "Atraining_murmur"
            parts = class_dir.name.split("_",1)
            folder = class_dir.name            # tüm klasör ismi
            label  = parts[1]                  # “murmur”, “normal”, vb.
            for wav in class_dir.glob("*.wav"):
                rows.append({
                    "path": str(wav),
                    "source": source,
                    "folder": folder,
                    "label": label
                })
    df = pd.DataFrame(rows)
    if df.empty:
        logger.error("Hiç kayıt bulunamadı!")
    return df
