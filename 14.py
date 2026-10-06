import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import rankdata, kruskal
from statsmodels.stats.multitest import multipletests

TREATED_ARM = "Treated"

d = tiles.loc[
    tiles["arm"].eq(TREATED_ARM)
].copy()

# Standardize binary labels.
d["tls_status"] = d["tls_status"].map({
    1: "TLS+",
    0: "TLS-",
    True: "TLS+",
    False: "TLS-",
    "positive": "TLS+",
    "negative": "TLS-",
    "TLS+": "TLS+",
    "TLS-": "TLS-",
})

d["outcome_pred"] = d["outcome_pred"].map({
    1: "Model+",
    0: "Model-",
    True: "Model+",
    False: "Model-",
    "positive": "Model+",
    "negative": "Model-",
    "Model+": "Model+",
    "Model-": "Model-",
})

d["patient_group"] = (
    d["tls_status"] + " / " + d["outcome_pred"]
)

########################################################################################################
########################################################################################################
GROUP_ORDER = [
    "TLS+ / Model+",
    "TLS+ / Model-",
    "TLS- / Model+",
    "TLS- / Model-",
]

d["patient_group"] = pd.Categorical(
    d["patient_group"],
    categories=GROUP_ORDER,
    ordered=True,
)

########################################################################################################
########################################################################################################

def calculate_patient_cluster_metrics(
    patient_tiles,
    min_cluster_tiles=5,
):
    patient_id = patient_tiles["patient_id"].iloc[0]

    N = len(patient_tiles)

    total_abs_contribution = (
        patient_tiles["signed_contribution"]
        .abs()
        .sum()
    )

    total_positive = (
        patient_tiles["signed_contribution"]
        .clip(lower=0)
        .sum()
    )

    total_negative = (
        -patient_tiles["signed_contribution"]
        .clip(upper=0)
        .sum()
    )

    rows = []

    for cluster_id, cluster_tiles in patient_tiles.groupby(
        "cluster_id"
    ):
        n_cluster = len(cluster_tiles)

        if n_cluster < min_cluster_tiles:
            continue

        prevalence = n_cluster / N

        tls_attention_mass = (
            cluster_tiles["tls_attention_normalized"]
            .sum()
        )

        # 0.5/N works as a small area-scale pseudocount.
        epsilon = 0.5 / N

        tls_mass_enrichment = (
            (tls_attention_mass + epsilon)
            / (prevalence + epsilon)
        )

        net_contribution = (
            cluster_tiles["signed_contribution"]
            .sum()
        )

        positive_contribution = (
            cluster_tiles["signed_contribution"]
            .clip(lower=0)
            .sum()
        )

        negative_contribution = (
            -cluster_tiles["signed_contribution"]
            .clip(upper=0)
            .sum()
        )

        normalized_net = (
            net_contribution / total_abs_contribution
            if total_abs_contribution > 0
            else np.nan
        )

        positive_fraction = (
            positive_contribution / total_positive
            if total_positive > 0
            else np.nan
        )

        negative_fraction = (
            negative_contribution / total_negative
            if total_negative > 0
            else np.nan
        )

        positive_sign_fraction = (
            cluster_tiles["signed_contribution"] > 0
        ).mean()

        rows.append({
            "patient_id": patient_id,
            "cluster_id": cluster_id,
            "n_cluster_tiles": n_cluster,
            "n_patient_tiles": N,
            "cluster_prevalence": prevalence,
            "tls_attention_mass": tls_attention_mass,
            "tls_mass_enrichment": tls_mass_enrichment,
            "log2_tls_enrichment": np.log2(
                tls_mass_enrichment
            ),
            "net_contribution": net_contribution,
            "normalized_net_contribution": normalized_net,
            "positive_contribution_fraction": positive_fraction,
            "negative_contribution_fraction": negative_fraction,
            "positive_sign_fraction": positive_sign_fraction,
        })

    return pd.DataFrame(rows)

########################################################################################################
########################################################################################################



def bootstrap_median(
    values,
    n_boot=2000,
    seed=2026,
):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    estimates = []

    for _ in range(n_boot):
        sample = rng.choice(
            values,
            size=len(values),
            replace=True,
        )
        estimates.append(np.median(sample))

    return (
        np.median(values),
        np.percentile(estimates, 2.5),
        np.percentile(estimates, 97.5),
    )


########################################################################################################
########################################################################################################

def summarize_cluster(
    cluster_data,
    total_patients,
):
    tls_mid, tls_low, tls_high = bootstrap_median(
        cluster_data["log2_tls_enrichment"]
    )

    out_mid, out_low, out_high = bootstrap_median(
        cluster_data["normalized_net_contribution"]
    )

    n_patients = cluster_data["patient_id"].nunique()

    return pd.Series({
        "n_patients": n_patients,
        "patient_prevalence": n_patients / total_patients,

        "median_log2_tls_enrichment": tls_mid,
        "tls_ci_low": tls_low,
        "tls_ci_high": tls_high,

        "median_outcome_direction": out_mid,
        "outcome_ci_low": out_low,
        "outcome_ci_high": out_high,

        "median_positive_sign_fraction": (
            cluster_data["positive_sign_fraction"]
            .median()
        ),

        "median_cluster_prevalence": (
            cluster_data["cluster_prevalence"]
            .median()
        ),

        "median_positive_contribution_fraction": (
            cluster_data[
                "positive_contribution_fraction"
            ].median()
        ),

        "median_negative_contribution_fraction": (
            cluster_data[
                "negative_contribution_fraction"
            ].median()
        ),
    })


    "TLS+ / Model+",
    "TLS+ / Model-",
    "TLS- / Model+",
    "TLS- / Model-",
]

d["patient_group"] = pd.Categorical(
    d["patient_group"],
    categories=GROUP_ORDER,
    ordered=True,
)


########################################################################################################
########################################################################################################


def assign_tls_class(row):
    if row["tls_ci_low"] > 0:
        return "TLS-related"

    if row["tls_ci_high"] < 0:
        return "TLS-unrelated"

    return "TLS-indeterminate"


def assign_outcome_class(row):
    if row["outcome_ci_low"] > 0:
        return "Outcome-positive"

    if row["outcome_ci_high"] < 0:
        return "Outcome-negative"

    return "Outcome-mixed"


cluster_summary["tls_class"] = (
    cluster_summary.apply(assign_tls_class, axis=1)
)

cluster_summary["outcome_class"] = (
    cluster_summary.apply(
        assign_outcome_class,
        axis=1,
    )
)

cluster_summary["candidate_class"] = (
    cluster_summary["tls_class"]
    + " / "
    + cluster_summary["outcome_class"]
)

########################################################################################################
########################################################################################################

fig, ax = plt.subplots(figsize=(10, 7))

sizes = (
    100
    + 1500 * cluster_summary["patient_prevalence"]
)

ax.scatter(
    cluster_summary["median_log2_tls_enrichment"],
    100 * cluster_summary["median_outcome_direction"],
    s=sizes,
    c=cluster_summary["plot_color"],
    edgecolor="black",
    linewidth=0.7,
    alpha=0.85,
)

# Patient-bootstrap confidence intervals.
for _, row in cluster_summary.iterrows():
    x = row["median_log2_tls_enrichment"]
    y = 100 * row["median_outcome_direction"]

    ax.errorbar(
        x,
        y,
        xerr=[[
            x - row["tls_ci_low"]
        ], [
            row["tls_ci_high"] - x
        ]],
        yerr=[[
            100 * (
                row["median_outcome_direction"]
                - row["outcome_ci_low"]
            )
        ], [
            100 * (
                row["outcome_ci_high"]
                - row["median_outcome_direction"]
            )
        ]],
        color="#555555",
        alpha=0.35,
        linewidth=0.8,
        zorder=0,
    )

ax.axvline(0, color="#444444", linewidth=1)
ax.axhline(0, color="#444444", linewidth=1)

ax.set_xlabel("Median log2 TLS attention enrichment")
ax.set_ylabel(
    "Median normalized outcome contribution (%)"
)

sns.despine()
plt.tight_layout()
plt.savefig(
    "slide14_cluster_quadrants.png",
    dpi=300,
    bbox_inches="tight",
)



label_mask = (
    (cluster_summary["patient_prevalence"] >= 0.10)
    & (
        cluster_summary["candidate_class"]
        .isin(QUADRANT_COLORS)
    )
)

for _, row in cluster_summary.loc[label_mask].iterrows():
    label = (
        row["pathology_label"]
        if pd.notna(row["pathology_label"])
        else f"Cluster {row['cluster_id']}"
    )

    ax.annotate(
        label,
        (
            row["median_log2_tls_enrichment"],
            100 * row["median_outcome_direction"],
        ),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8,
    )

########################################################################################################
########################################################################################################