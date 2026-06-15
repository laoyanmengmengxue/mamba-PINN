from pathlib import Path
import json
import pandas as pd


ROOT = Path(__file__).resolve().parents[2] / "results" / "revision_studies" / "baselines" / "gru_tcn"


def main() -> None:
    rows = []
    for case in ["single", "triple"]:
        for model in ["gru", "tcn"]:
            d = ROOT / case / model
            metrics_path = d / "metrics_by_window.csv"
            meta_path = d / "metadata.json"
            if not metrics_path.exists():
                print(f"MISSING {metrics_path}")
                continue
            df = pd.read_csv(metrics_path)
            row = df[df["window"] == "extrapolation_5_7s"].iloc[0].to_dict()
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                row.update(
                    {
                        "train_seconds": meta.get("train_seconds"),
                        "n_points_used": meta.get("n_points_used"),
                        "n_points_total_file": meta.get("n_points_total_file"),
                        "input_len": meta.get("input_len"),
                        "pred_len": meta.get("pred_len"),
                        "epochs": meta.get("epochs"),
                        "max_batches_per_epoch": meta.get("max_batches_per_epoch"),
                    }
                )
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "combined_gru_tcn_metrics_summary.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
