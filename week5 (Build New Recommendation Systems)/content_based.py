# a purely content-based recommender
# item feature vectors are from content_metadata_enriched.parquet
# computes item to item nearest neighbors using cosine similarity
# scoring for each user done based on neighbors of items in their qualified TRAIN history
# note that users with no eligible train history will receive no recommendations

import duckdb
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# helper for readable console logs
def print_section_header(title, no_endline=False):
    print("\n" + "=" * 95)
    print(title)
    print("=" * 95)
    if not no_endline:
        print()

# CONFIG VARIABLES

TOP_K_RECS = 10

QUALIFIED_THRESHOLD = 0.60

ITEM_NEIGHBOR_K = 100  # item->item neighbor count

TRAIN_SET_PATH = "data/data-parquet-files/test-train-splits/content_views_train_split.parquet"
TEST_SET_PATH = "data/data-parquet-files/test-train-splits/content_views_test_split.parquet"

CONTENT_METADATA_PATH = "data/data-parquet-files/content_metadata_enriched.parquet"

OUT_DIR = "data/recommender-output-files"

# outputs
NEIGHBORS_OUT_PATH = f"{OUT_DIR}/content_item_neighbors_top{ITEM_NEIGHBOR_K}.parquet"
RECS_CONTENT_ALL_TEST_PATH = f"{OUT_DIR}/recs_content_all_test_users_top{TOP_K_RECS}.parquet"
RECS_CONTENT_WARM_TEST_PATH = f"{OUT_DIR}/recs_content_warm_test_users_top{TOP_K_RECS}.parquet"

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# load content metadata

con = duckdb.connect()

print_section_header("1) Loading content metadata")

content_df = con.execute(f"""
    select
        content_id,
        studio,
        title,
        genre_id,
        minutes,
        language_code,
        duration_seconds
    from read_parquet('{CONTENT_METADATA_PATH}')
""").df()

# basic cleanup so no issue with nulls
for col in ["studio", "title", "genre_id", "language_code"]:
    content_df[col] = content_df[col].fillna("unknown").astype(str)

for col in ["minutes", "duration_seconds"]:
    if content_df[col].isna().any():
        med = float(content_df[col].dropna().median()) if content_df[col].dropna().shape[0] > 0 else 0.0
        content_df[col] = content_df[col].fillna(med)
    content_df[col] = content_df[col].astype(float)

# ensure content_id is string-like
content_df["content_id"] = content_df["content_id"].astype(str)

print(f"loaded {len(content_df):,} items from metadata")
print("\nmetadata sample:")
print(content_df.head(10).to_string(index=False))

# build item feature matrix (title tf-idf, categorical one-hot, numerics scaled)

print_section_header("2) Building item feature matrix")

text_feature = "title"
categorical_features = ["studio", "genre_id", "language_code"]
numeric_features = ["minutes", "duration_seconds"]

# column transformer that outputs a sparse feature matrix
feature_builder = ColumnTransformer(
    transformers=[
        ("title_tfidf", TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            min_df=2,
            token_pattern=r"(?u)\b\w+\b"
        ), text_feature),

        ("cats", OneHotEncoder(handle_unknown="ignore"), categorical_features),

        # standardscaler(with_mean=False) keeps it compatible with sparse matrices
        ("nums", Pipeline(steps=[
            ("scaler", StandardScaler(with_mean=False))
        ]), numeric_features),
    ],
    remainder="drop",
    sparse_threshold=0.3
)

X = feature_builder.fit_transform(content_df)

print(f"feature matrix shape: {X.shape[0]:,} items x {X.shape[1]:,} features")

# fit item->item neighbors using cosine

print_section_header(f"3) Fitting NearestNeighbors (ITEM_NEIGHBOR_K = {ITEM_NEIGHBOR_K})")

# cosine distance returned by sklearn = 1 - cosine_similarity
nn = NearestNeighbors(
    n_neighbors=ITEM_NEIGHBOR_K + 1,  # +1 because the nearest neighbor is the item itself
    metric="cosine",
    algorithm="brute",
    n_jobs=-1
)

nn.fit(X)

distances, indices = nn.kneighbors(X, return_distance=True)  # get distances

content_ids = content_df["content_id"].tolist()

# build a neighbors dataframe
# for each item, list its top k neighbors excluding itself
rows = []
for i, (nbr_idx_row, dist_row) in enumerate(zip(indices, distances)):
    src_id = content_ids[i]

    # skip the first neighbor if it's itself (it should be with cosine)
    for rank_pos in range(1, min(ITEM_NEIGHBOR_K + 1, len(nbr_idx_row))):
        j = int(nbr_idx_row[rank_pos])
        dist = float(dist_row[rank_pos])
        sim = 1.0 - dist

        # keep only positive similarity
        if sim <= 0.0:
            continue

        rows.append({
            "content_id": src_id,
            "neighbor_content_id": content_ids[j],
            "neighbor_rank": rank_pos,
            "similarity": float(sim),
        })

neighbors_df = pd.DataFrame(rows)

print(f"built {len(neighbors_df):,} neighbor edges")
print("\nneighbor sample:")
print(neighbors_df.head(15).to_string(index=False))

con.register("neighbors_df", neighbors_df)
con.execute(f"copy neighbors_df to '{NEIGHBORS_OUT_PATH}' (format parquet)")
print(f"\nsaved neighbor table to: {NEIGHBORS_OUT_PATH}")

# build warm/cold user sets from train test splits

print_section_header("4) Discovering test users and warm test users (train intersect test)")

train_users = con.execute(f"""
    select distinct adventurer_id
    from read_parquet('{TRAIN_SET_PATH}')
""").df()["adventurer_id"].tolist()

test_users = con.execute(f"""
    select distinct adventurer_id
    from read_parquet('{TEST_SET_PATH}')
""").df()["adventurer_id"].tolist()

train_users_set = set(train_users)
test_users_set = set(test_users)
warm_test_users = sorted(train_users_set.intersection(test_users_set))

print(f"train users                                 = {len(train_users_set):,}")
print(f"test users                                  = {len(test_users_set):,}")
print(f"warm test users (are in both train and test) = {len(warm_test_users):,}")

# build warm user history from train (threshold-filtered)

print_section_header("5) Building warm user history from TRAIN (threshold-filtered)")

warm_hist_df = con.execute(f"""
    select
        adventurer_id,
        list(distinct content_id) as hist_items
    from read_parquet('{TRAIN_SET_PATH}')
    where watch_ratio >= {QUALIFIED_THRESHOLD}
      and adventurer_id in (select unnest(?))
    group by adventurer_id
""", [warm_test_users]).df()

warm_user_to_hist = dict(zip(warm_hist_df["adventurer_id"], warm_hist_df["hist_items"]))

print(f"built histories for {len(warm_user_to_hist):,} warm users using watch_ratio >= {QUALIFIED_THRESHOLD}")
print("\nsample history sizes:")
tmp = warm_hist_df.head(10).copy()
tmp["hist_count"] = tmp["hist_items"].apply(lambda x: 0 if x is None else len(x))
print(tmp[["adventurer_id", "hist_count"]].to_string(index=False))

# build warm user -> seen-items lookup from train (excluding repeats)

print_section_header("6) Building warm user")

warm_seen_df = con.execute(f"""
    select
        adventurer_id,
        list(distinct content_id) as seen_items
    from read_parquet('{TRAIN_SET_PATH}')
    where adventurer_id in (select unnest(?))
    group by adventurer_id
""", [warm_test_users]).df()

warm_user_to_seen = dict(zip(warm_seen_df["adventurer_id"], warm_seen_df["seen_items"]))

print(f"built seen-items lists for {len(warm_user_to_seen):,} warm users")

# build neighbor dict for fast scoring

# neighbor_map[src_item] -> list of (neighbor_item, similarity)
neighbor_map = {}
for row in neighbors_df.itertuples(index=False):
    neighbor_map.setdefault(row.content_id, []).append((row.neighbor_content_id, float(row.similarity)))

# score users

print_section_header(f"7) Scoring and saving recommendations (TOP_K_RECS = {TOP_K_RECS})")

def top_k_content_based_for_user(user_id, topk):
    # if user has no eligible history, we return no recs
    hist_items = warm_user_to_hist.get(user_id, None)
    if hist_items is None or len(hist_items) == 0:
        return []

    seen_items = warm_user_to_seen.get(user_id, None)
    seen_set = set(seen_items) if seen_items is not None else set()

    # aggregate neighbor similarities from each history item
    scores = {}

    for src in set(hist_items):
        src = str(src)
        for nbr, sim in neighbor_map.get(src, []):
            if nbr in seen_set:
                continue
            scores[nbr] = scores.get(nbr, 0.0) + sim

    if not scores:
        return []

    # take top-k by aggregated similarity
    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topk]
    return [(cid, float(score)) for cid, score in top]

def score_user_list(users, label):
    print(f" SCORING USERS: {label}\n")
    rows = []
    users_with_recs = 0

    for i, u in enumerate(users):
        top_recs = top_k_content_based_for_user(u, topk=TOP_K_RECS)

        if not top_recs:
            continue

        users_with_recs += 1
        for rank, (cid, score) in enumerate(top_recs, start=1):
            rows.append({
                "adventurer_id": u,
                "rec_rank": rank,
                "content_id": str(cid),
                "score": float(score),
            })

        if (i + 1) % 5000 == 0:
            print(f"processed {i + 1:,} users...")

    rec_df = pd.DataFrame(rows)
    print(f"\nusers with recs = {users_with_recs:,} / {len(users):,}")
    print(f"produced {len(rec_df):,} recommendation rows")
    return rec_df

# all test users
recs_all_test_df = score_user_list(sorted(test_users_set), "ALL TEST USERS (CONTENT-BASED)")
con.register("recs_all_test_df", recs_all_test_df)
con.execute(f"copy recs_all_test_df to '{RECS_CONTENT_ALL_TEST_PATH}' (format parquet)")
print(f"\nsaved all-test-user recs to: {RECS_CONTENT_ALL_TEST_PATH}\n")

# warm test users
recs_warm_test_df = score_user_list(warm_test_users, "WARM TEST USERS (CONTENT-BASED)")
con.register("recs_warm_test_df", recs_warm_test_df)
con.execute(f"copy recs_warm_test_df to '{RECS_CONTENT_WARM_TEST_PATH}' (format parquet)")
print(f"\nsaved warm-test-user recs to: {RECS_CONTENT_WARM_TEST_PATH}")

# Samples for sanity check

print_section_header("9) Sample recommendations")

print("content-based all-test-users sample...")
print(recs_all_test_df.head(20).to_string(index=False))

print("\ncontent-based warm-test-users sample...")
print(recs_warm_test_df.head(20).to_string(index=False))

print_section_header("DONE.", no_endline=True)

con.close()