import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

OUTPUT_DIR = "eda_outputs"
TARGET_COLUMNS = [
    "Total Alkalinity",
    "Electrical Conductance",
    "Dissolved Reactive Phosphorus",
]


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_and_merge_training_data() -> pd.DataFrame:
    water_quality_df = pd.read_csv("water_quality_training_dataset.csv")
    landsat_train = pd.read_csv("landsat_features_training.csv")
    climate_train = pd.read_csv("climate_features_training.csv")

    for col in ["NDMI", "MNDWI"]:
        if col in landsat_train.columns:
            landsat_train[col] = pd.to_numeric(landsat_train[col], errors="coerce")

    data = pd.concat([water_quality_df, landsat_train, climate_train], axis=1)
    data = data.loc[:, ~data.columns.duplicated()]

    if "Sample Date" in data.columns:
        data["Sample Date"] = pd.to_datetime(data["Sample Date"], format="%d-%m-%Y", errors="coerce")

    return data


def save_overview(df: pd.DataFrame) -> None:
    summary_lines = []
    summary_lines.append("=== EDA OVERVIEW ===")
    summary_lines.append(f"Rows: {df.shape[0]}")
    summary_lines.append(f"Columns: {df.shape[1]}")
    summary_lines.append("")

    missing_total = int(df.isna().sum().sum())
    total_values = int(df.shape[0] * df.shape[1])
    missing_pct = (missing_total / total_values * 100.0) if total_values else 0.0

    summary_lines.append(f"Missing total: {missing_total} ({missing_pct:.2f}%)")
    summary_lines.append(f"Duplicated rows: {int(df.duplicated().sum())}")
    summary_lines.append("")

    summary_lines.append("Dtypes:")
    for k, v in df.dtypes.astype(str).to_dict().items():
        summary_lines.append(f"- {k}: {v}")

    overview_path = os.path.join(OUTPUT_DIR, "eda_overview.txt")
    with open(overview_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))


def save_missing_value_report(df: pd.DataFrame) -> None:
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]

    missing_df = pd.DataFrame(
        {
            "column": missing.index,
            "missing_count": missing.values,
            "missing_pct": (missing.values / len(df) * 100.0),
        }
    )

    missing_df.to_csv(os.path.join(OUTPUT_DIR, "missing_values_report.csv"), index=False)


def save_numeric_describe(df: pd.DataFrame) -> pd.DataFrame:
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty:
        return num_df

    describe_df = num_df.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    describe_df.to_csv(os.path.join(OUTPUT_DIR, "numeric_describe.csv"))
    return num_df


def save_target_plots(df: pd.DataFrame) -> None:
    present_targets = [t for t in TARGET_COLUMNS if t in df.columns]
    if not present_targets:
        return

    for target in present_targets:
        s = pd.to_numeric(df[target], errors="coerce").dropna()
        if s.empty:
            continue

        fig, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.hist(s.values, bins=40)
        ax.set_title(f"Histogram - {target}")
        ax.set_xlabel(target)
        ax.set_ylabel("Frequency")

        fig.tight_layout()
        out_file = f"target_distribution_{target.replace(' ', '_')}.png"
        fig.savefig(os.path.join(OUTPUT_DIR, out_file), dpi=150, bbox_inches="tight")
        plt.close(fig)


def iqr_outlier_summary(num_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in num_df.columns:
        s = pd.to_numeric(num_df[col], errors="coerce").dropna()
        if len(s) < 5:
            continue

        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            out_count = 0
            lower = q1
            upper = q3
        else:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            out_count = int(((s < lower) | (s > upper)).sum())

        pct = out_count / len(s) * 100.0
        rows.append(
            {
                "column": col,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower,
                "upper_bound": upper,
                "outlier_count": out_count,
                "outlier_pct": pct,
            }
        )

    outlier_df = pd.DataFrame(rows).sort_values("outlier_pct", ascending=False)
    outlier_df.to_csv(os.path.join(OUTPUT_DIR, "outliers_iqr_report.csv"), index=False)
    return outlier_df


def save_correlation_reports(num_df: pd.DataFrame, full_df: pd.DataFrame) -> None:
    if num_df.empty:
        return

    corr = num_df.corr(numeric_only=True)
    corr.to_csv(os.path.join(OUTPUT_DIR, "correlation_matrix_numeric.csv"))

    present_targets = [t for t in TARGET_COLUMNS if t in full_df.columns and t in corr.columns]
    if present_targets:
        target_corr = corr[present_targets].copy()
        target_corr["max_abs_target_corr"] = target_corr.abs().max(axis=1)
        target_corr = target_corr.sort_values("max_abs_target_corr", ascending=False)
        target_corr.to_csv(os.path.join(OUTPUT_DIR, "target_correlation_report.csv"))

    variances = num_df.var(numeric_only=True).sort_values(ascending=False)
    top_cols = variances.head(25).index.tolist()
    top_corr = num_df[top_cols].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(top_corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(top_cols)))
    ax.set_xticklabels(top_cols, rotation=90, fontsize=8)
    ax.set_yticks(range(len(top_cols)))
    ax.set_yticklabels(top_cols, fontsize=8)
    ax.set_title("Correlation Heatmap (Top 25 Numeric Features by Variance)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap_top25.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_time_profile(df: pd.DataFrame) -> None:
    if "Sample Date" not in df.columns:
        return

    s = pd.to_datetime(df["Sample Date"], errors="coerce").dropna()
    if s.empty:
        return

    profile = (
        pd.DataFrame({"Sample Date": s})
        .assign(year=lambda x: x["Sample Date"].dt.year, month=lambda x: x["Sample Date"].dt.month)
        .groupby(["year", "month"])
        .size()
        .reset_index(name="count")
    )
    profile.to_csv(os.path.join(OUTPUT_DIR, "time_profile_year_month.csv"), index=False)


def main() -> None:
    ensure_output_dir(OUTPUT_DIR)

    df = load_and_merge_training_data()

    print("Running EDA...")
    print(f"Dataset shape: {df.shape}")

    save_overview(df)
    save_missing_value_report(df)

    num_df = save_numeric_describe(df)
    save_target_plots(df)
    outlier_df = iqr_outlier_summary(num_df)
    save_correlation_reports(num_df, df)
    save_time_profile(df)

    print("EDA completed. Outputs saved in: eda_outputs")
    if not outlier_df.empty:
        print("Top 10 columns by outlier percentage (IQR):")
        print(outlier_df[["column", "outlier_pct"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
