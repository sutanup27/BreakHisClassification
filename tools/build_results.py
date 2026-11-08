"""Build an offline results snapshot from saved reports; never fits/evaluates models."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def collect(root):
    records, warnings = [], []
    for run in sorted(root.glob("*")):
        if not run.is_dir() or not (run / "config.json").exists():
            continue
        try:
            config = read(run / "config.json")
            data = config["data"]
            # Classify by the configured data path, not an attractive run name.
            path = data.get("data_root", "").replace("\\", "/").lower()
            category = "verification" if "/artifacts/" in path else "real" if path else "unknown"
            for phase, folder in (("legacy", run), ("development", run / "development"), ("final", run / "final")):
                for report in sorted(folder.glob("*.metrics.json")):
                    try:
                        m = read(report)
                        im, gm = m.get("image_metrics", {}), m.get("group_metrics", {})
                        benchmark = m.get("benchmark", {})
                        history = []
                        for name in ("train", "finetune"):
                            h = run / (name + ".history.json")
                            if h.exists():
                                history.extend(dict(row, stage=name) for row in read(h).get("epochs", []))
                        rel = str(report.relative_to(root)).replace("\\", "/")
                        records.append(dict(id=run.name + "/" + phase + "/" + report.stem,
                            run=run.name, category=category, phase=phase, role=report.name.removesuffix(".metrics.json"),
                            split=m.get("split", "unknown"), model=config["training"]["model"],
                            magnification=data.get("magnification"), baseline=config.get("baseline", {}).get("kind", "standard"),
                            ranking=config.get("baseline", {}).get("ranking"), seed=config["training"].get("seed"),
                            accuracy=im.get("accuracy", m.get("accuracy")), sensitivity=im.get("sensitivity"),
                            specificity=im.get("specificity"), group_balanced=gm.get("balanced_accuracy"),
                            samples=im.get("samples"), groups=gm.get("samples"), matrix=im.get("confusion_matrix"),
                            classes=im.get("classes", ["benign", "malignant"]),
                            latency=benchmark.get("latency_ms", {}).get("median"),
                            device=benchmark.get("runtime", {}).get("device"),
                            macs=benchmark.get("conv_linear_macs_per_image"),
                            config=config, history=history, report=rel))
                    except (ValueError, KeyError, TypeError, OSError) as error:
                        warnings.append(str(report) + ": " + str(error))
        except (ValueError, KeyError, TypeError, OSError) as error:
            warnings.append(str(run) + ": " + str(error))
    return records, warnings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "artifacts/experiments")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/results.html")
    args = parser.parse_args()
    if not args.runs.is_dir():
        parser.error("Saved runs directory does not exist: " + str(args.runs))
    records, warnings = collect(args.runs)
    payload = dict(generated=datetime.now(timezone.utc).isoformat(), records=records, warnings=warnings)
    # Keep even hostile report text inside an inert JSON block.
    data = json.dumps(payload, allow_nan=False).replace("<", "\\u003c").replace("&", "\\u0026")
    template = (ROOT / "tools/results_template.html").read_text(encoding="utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(template.replace("__RESULTS_DATA__", data), encoding="utf-8")
    print(f"Built {args.output}: {len(records)} saved reports; {len(warnings)} warnings")


if __name__ == "__main__":
    main()
