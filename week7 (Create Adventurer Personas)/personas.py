from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("data/data-parquet-files")
OUT_DIR = Path("outputs/clustering_k5")

K = 5  # fixed k chosen via elbow

RANDOM_STATE = 42  # for k-means reproducibility


def read_parquet(path: Path, columns: list[str]) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(f"missing parquet file: {path}")
    return pd.read_parquet(path, columns=columns)


def build_adventurer_features() -> pd.DataFrame:

    views_path = DATA_DIR / "content_views_enriched.parquet"
    subs_path = DATA_DIR / "subscriptions_enriched.parquet"
    opens_path = DATA_DIR / "app_opens.parquet"

    views = read_parquet(
        views_path,
        columns=[
            "adventurer_id",
            "publisher_id",
            "playlist_id",
            "seconds_viewed",
            "watch_ratio",
            "event_day_index",
        ],
    )

    subs = read_parquet(
        subs_path,
        columns=[
            "adventurer_id",
            "publisher_id",
            "event_day_index",
        ],
    )

    opens = read_parquet(
        opens_path,
        columns=[
            "adventurer_id",
            "publisher_id",
            "playlist_id",
        ],
    )

    # computing a "today" anchor for recency (max day seen in event-indexed tables)
    max_view_day = views["event_day_index"].max() if len(views) else np.nan
    max_sub_day = subs["event_day_index"].max() if len(subs) else np.nan
    global_max_day = np.nanmax([max_view_day, max_sub_day])

    if np.isnan(global_max_day):
        raise ValueError("could not determine global_max_day (event_day_index missing/empty).")

    # content views aggregations
    views_agg = (
        views.groupby("adventurer_id", as_index=True)
        .agg(
            views_total=("adventurer_id", "size"),
            seconds_viewed_total=("seconds_viewed", "sum"),
            watch_ratio_mean=("watch_ratio", "mean"),
            active_days_views=("event_day_index", "nunique"),
            unique_publishers_viewed=("publisher_id", "nunique"),
            unique_playlists_viewed=("playlist_id", "nunique"),
            last_view_day_index=("event_day_index", "max"),
        )
    )
    views_agg["days_since_last_view"] = global_max_day - views_agg["last_view_day_index"]
    views_agg = views_agg.drop(columns=["last_view_day_index"])

    # subscriptions aggregations
    subs_agg = (
        subs.groupby("adventurer_id", as_index=True)
        .agg(
            subs_total=("adventurer_id", "size"),
            unique_publishers_subbed=("publisher_id", "nunique"),
            last_sub_day_index=("event_day_index", "max"),
        )
    )
    subs_agg["days_since_last_sub"] = global_max_day - subs_agg["last_sub_day_index"]
    subs_agg = subs_agg.drop(columns=["last_sub_day_index"])

    # app opens aggregations
    opens_agg = (
        opens.groupby("adventurer_id", as_index=True)
        .agg(
            app_opens_total=("adventurer_id", "size"),
            unique_publishers_opened=("publisher_id", "nunique"),
            unique_playlists_opened=("playlist_id", "nunique"),
        )
    )

    # merge into one table
    features = views_agg.join(subs_agg, how="outer").join(opens_agg, how="outer")

    # fill missing values
    features["watch_ratio_mean"] = features["watch_ratio_mean"].fillna(0.0)

    count_like_cols = [
        "views_total",
        "seconds_viewed_total",
        "active_days_views",
        "unique_publishers_viewed",
        "unique_playlists_viewed",
        "subs_total",
        "unique_publishers_subbed",
        "app_opens_total",
        "unique_publishers_opened",
        "unique_playlists_opened",
    ]
    for col in count_like_cols:
        if col in features.columns:
            features[col] = features[col].fillna(0.0)

    never_value = float(global_max_day + 1)
    for col in ["days_since_last_view", "days_since_last_sub"]:
        if col in features.columns:
            features[col] = features[col].fillna(never_value)

    # ensure numeric columns are numeric
    for col in features.columns:
        features[col] = pd.to_numeric(features[col], errors="coerce").fillna(0.0)

    # keep id as a column
    features = features.reset_index()  # adventurer_id becomes a column

    return features


# preprocess:
# log1p on heavy-tailed nonnegative features
# standardize for kmeans
# returns:
# numpy array used to fit kmeans
def preprocess_for_kmeans(features: pd.DataFrame) -> tuple[np.ndarray, list[str]]:

    X_df = features.drop(columns=["adventurer_id"]).copy()

    log1p_cols = [
        "views_total",
        "seconds_viewed_total",
        "active_days_views",
        "unique_publishers_viewed",
        "unique_playlists_viewed",
        "subs_total",
        "unique_publishers_subbed",
        "app_opens_total",
        "unique_publishers_opened",
        "unique_playlists_opened",
        "days_since_last_view",
        "days_since_last_sub",
    ]
    for col in log1p_cols:
        if col in X_df.columns:
            X_df[col] = np.log1p(np.clip(X_df[col].to_numpy(dtype=float), a_min=0.0, a_max=None))

    scaler = StandardScaler()
    X = scaler.fit_transform(X_df.to_numpy(dtype=float))

    return X, list(X_df.columns)


def write_cluster_summaries(features_with_cluster: pd.DataFrame) -> None:

    # cluster sizes
    sizes = (
        features_with_cluster.groupby("cluster", as_index=False)
        .agg(cluster_size=("adventurer_id", "size"))
        .sort_values("cluster")
    )
    sizes.to_csv(OUT_DIR / "cluster_sizes.csv", index=False)

    # means and medians (exclude adventurer_id)
    numeric_cols = [c for c in features_with_cluster.columns if c not in ["adventurer_id", "cluster"]]
    means = (
        features_with_cluster.groupby("cluster")[numeric_cols]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values("cluster")
    )
    medians = (
        features_with_cluster.groupby("cluster")[numeric_cols]
        .median(numeric_only=True)
        .reset_index()
        .sort_values("cluster")
    )

    means.to_csv(OUT_DIR / "cluster_summary_mean.csv", index=False)
    medians.to_csv(OUT_DIR / "cluster_summary_median.csv", index=False)


def main() -> None:

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # build features
    features = build_adventurer_features()
    features.to_csv(OUT_DIR / "adventurer_features.csv", index=False)

    # preprocess and fit final kmeans
    X, feature_cols = preprocess_for_kmeans(features)

    km = KMeans(
        n_clusters=K,
        random_state=RANDOM_STATE,
        n_init=10,  # avoids sklearn version issues vs n_init="auto"
    )
    clusters = km.fit_predict(X)

    # write assignments
    assignments = features[["adventurer_id"]].copy()
    assignments["cluster"] = clusters
    assignments.to_csv(OUT_DIR / "cluster_assignments.csv", index=False)

    # attach cluster labels to feature table for summaries
    features_with_cluster = features.copy()
    features_with_cluster["cluster"] = clusters

    # write summary tables
    write_cluster_summaries(features_with_cluster)

    print(f"wrote: {OUT_DIR / 'adventurer_features.csv'}")
    print(f"wrote: {OUT_DIR / 'cluster_assignments.csv'}")
    print(f"wrote: {OUT_DIR / 'cluster_sizes.csv'}")
    print(f"wrote: {OUT_DIR / 'cluster_summary_mean.csv'}")
    print(f"wrote: {OUT_DIR / 'cluster_summary_median.csv'}")


if __name__ == "__main__":
    main()
