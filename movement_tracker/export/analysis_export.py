"""CSV time series plus a versioned JSON sidecar with full analysis metadata."""
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
import csv
import hashlib
import json
import os
import shutil
import numpy as np
from movement_tracker.analysis.models import EVENT_NAMES

def export_analysis(result, csv_path):
    csv_path = Path(csv_path).with_suffix(".csv")
    json_path = csv_path.with_suffix(".json")
    source = result.metadata.get("source_path")
    if source and Path(source).resolve() in (csv_path.resolve(), json_path.resolve()):
        raise ValueError("Analysis exports cannot overwrite their source.")
    times = np.asarray(result.source_timestamps)
    event_rows = {name: None if event.source_timestamp is None else int(np.argmin(np.abs(times - event.source_timestamp)))
                  for name, event in result.events.items()}
    with TemporaryDirectory(prefix=".analysis-export-", dir=csv_path.parent) as temporary:
        temporary = Path(temporary)
        staged_csv, staged_json = temporary / "new.csv", temporary / "new.json"
        with staged_csv.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["source_timestamp", "frame", *result.series, "quality_flags", *[f"event_{name}" for name in EVENT_NAMES]])
            for i, (timestamp, frame) in enumerate(zip(result.source_timestamps, result.frame_indices)):
                writer.writerow([timestamp, frame, *[values[i] for values in result.series.values()],
                                 "|".join(result.quality_flags[i]), *[int(event_rows[name] == i) for name in EVENT_NAMES]])
        payload = asdict(result)
        payload["export"] = {"csv_filename": csv_path.name,
                             "csv_sha256": hashlib.sha256(staged_csv.read_bytes()).hexdigest(),
                             "event_flags": "nearest sampled frame; JSON source_timestamp remains authoritative",
                             "event_nearest_frames": {name: None if index is None else result.frame_indices[index]
                                                      for name, index in event_rows.items()}}
        staged_json.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        # Stage both first; restore previous files if either final replacement fails.
        backups = {}
        replaced = []
        try:
            for target in (csv_path, json_path):
                if target.exists():
                    backup = temporary / ("old" + target.suffix)
                    shutil.copy2(target, backup)
                    backups[target] = backup
            for staged, target in ((staged_csv, csv_path), (staged_json, json_path)):
                os.replace(staged, target)
                replaced.append(target)
        except BaseException:
            for target in reversed(replaced):
                if target in backups:
                    os.replace(backups[target], target)
                else:
                    target.unlink(missing_ok=True)
            raise
    return csv_path, json_path
