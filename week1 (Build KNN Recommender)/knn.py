# builds a qualified-only item-item KNN CF for recommending items to users
# can run same eval driver (but change RECS_PATH = ".../recs_qualified_warm_test_users_top{K}.parquet")

# Some very important notes about this model...
#
# This is an “implicit” recommender; strength = count(qualified views) is the signal where qualified views meet a threshold.
#
# The model only recommends items that appear in the qualified train set.
# If an item never appears in qualified train, it cannot be recommended.
#
# Warm test users only: users without qualified train history get zero recs.
# Here “warm” means they appear in both the train and test split sets (so the model can score them).
# If they appeared in one but not the other, then they either have no offline-testing value or they have no history to recommend from (are cold-start users).

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from collections import defaultdict
import joblib

# helper for readable console logs
def print_section_header(title, no_endline=False):
    print("\n" + "=" * 95)
    print(title)
    print("=" * 95)
    if not no_endline:
        print()

# CONFIG VARIABLES

QUALIFIED_THRESHOLD = 0.60  # can change this at anytime (e.g. 0.20, 0.40)
TOP_K_RECS = 10  # how many recommendations outputted per user
K_NEIGHBORS = 50  # number of nearest neighbors per item for KNN

TRAIN_SET_PATH = "data/data-parquet-files/test-train-splits/content_views_train_split.parquet"
TEST_SET_PATH = "data/data-parquet-files/test-train-splits/content_views_test_split.parquet"

OUT_DIR = "data/recommender-output-files"

# outputs
TRAIN_QUALIFIED_PATH = f"{OUT_DIR}/content_views_train_split_qualified_only.parquet"
NEIGHBORS_QUAL_PATH = f"{OUT_DIR}/qualified_item_neighbors.joblib"
RECS_ALL_QUAL_PATH = f"{OUT_DIR}/recs_all_qualified_train_users_top{TOP_K_RECS}.parquet"
RECS_TEST_QUAL_PATH = f"{OUT_DIR}/recs_qualified_warm_test_users_top{TOP_K_RECS}.parquet"

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)  # ensure output directory exists

# -------------------------------------------------------------------
# 1) Create "qualified"-only TRAIN parquet
# -------------------------------------------------------------------

con = duckdb.connect()

print_section_header("1) Writing \"qualified\"-only TRAIN parquet file")

# create new content views file which fiters out content viewed under the threshold
con.execute(f"""
    copy (
        select *
        from read_parquet('{TRAIN_SET_PATH}')
        where watch_ratio >= {QUALIFIED_THRESHOLD}
    )
    to '{TRAIN_QUALIFIED_PATH}' (format parquet)
""")

# gather and output basic stats on qualified views file
qual_train_counts = con.execute(f"""
    select
        count(*) as qualified_train_rows,
        count(distinct adventurer_id) as qualified_train_users,
        count(distinct content_id) as qualified_train_items
    from read_parquet('{TRAIN_QUALIFIED_PATH}')
""").df()

print(f"qualified threshold used is watch_ratio >= {QUALIFIED_THRESHOLD}", '\n')
print(f"wrote qualified TRAIN to: {TRAIN_QUALIFIED_PATH}")
print(qual_train_counts.to_string(index=False))

# -------------------------------------------------------------------
# 2) Aggregate qualified interactions with frequency strength
# -------------------------------------------------------------------

print_section_header("2) Aggregating qualified training interactions by frequency viewed (strength)")

# collapses raw qualified views into a single row per (user,item) pair
# rows in this df are of the form (adv_id, content_id, strength)
# strength is simple implicit-feedback signal here meaning:
# "how many qualified views did this user have for this item?"
strength_df = con.execute(f"""
    select
        adventurer_id,
        content_id,
        count(*) as strength
    from read_parquet('{TRAIN_QUALIFIED_PATH}')
    group by adventurer_id, content_id
""").df()

print(f"collapsed into {len(strength_df):,} aggregated rows (user,item)")
print("\nsample of the new dataframe...")
print(strength_df.head(10).to_string(index=False))

# --------------------------------------------------------------------
# 3) Build mappings + CSR matrix from the qualified-views strength_df
# --------------------------------------------------------------------

print_section_header("3) Building the CSR user-item matrix of qualified views and their strengths")

# collect all user and item ids
user_ids = strength_df["adventurer_id"].unique()
item_ids = strength_df["content_id"].unique()

# map real ids to indices
user_id_to_idx = {u: i for i, u in enumerate(user_ids)}
item_id_to_idx = {c: j for j, c in enumerate(item_ids)}

# user and item counts
qualified_users_count = len(user_ids)
qualified_items_count = len(item_ids)

# prep other CSR matrix arguments
row_idx = strength_df["adventurer_id"].map(user_id_to_idx).to_numpy()
col_idx = strength_df["content_id"].map(item_id_to_idx).to_numpy()
data = strength_df["strength"].astype(np.float32).to_numpy()

# creates a sparse matrix of qualified U x I where entry (u,i) = strength
# CSR is used because efficient for row slicing (users) and general sparse storage
X_train_qualified = csr_matrix((data, (row_idx, col_idx)), shape=(qualified_users_count, qualified_items_count))

# calculate and output basic matrix stats
density = X_train_qualified.nnz / (
            qualified_users_count * qualified_items_count) if qualified_users_count > 0 and qualified_items_count > 0 else 0.0
sparsity = 1.0 - density

print(f"X_train_qualified shape = {X_train_qualified.shape}")
print(f"X_train_qualified nnz   = {X_train_qualified.nnz:,}")
print(f"density  = {density:.8f}")
print(f"sparsity = {sparsity:.8f}")

# ---------------------------------------------------------------------
# 4) Create item-item KNN neighbors from the qualified-only CSR matrix
# ---------------------------------------------------------------------

print_section_header("4) Fitting the item-item KNN")

# transpose so each item becomes a vector over users
X_item_user = X_train_qualified.T.tocsr()  # now items x users with shape of (I, U)

# build neighborhood model and fit on training data
knn = NearestNeighbors(
    n_neighbors=K_NEIGHBORS + 1,  # +1 for self
    metric="cosine",
    algorithm="brute",
    n_jobs=-1
)
knn.fit(X_item_user)  # “training” here just means store the vectors (because KNN is lazy)

distances, indices = knn.kneighbors(X_item_user, return_distance=True)  # for each item vector, find its nearest items
similarity_scores = 1.0 - distances  # similarity score is 1 - cosine distance

# build the neighbors data structure as a dict keyed by the actual content_id (skips self-neighbor)
# result is a dictionary mapping: item_id -> a list of its K_NEIGHBORS i.e. (neighbor_item_id, similarity)
item_id_to_item_neighbors = {}
for item_col_idx, (nbr_cols, nbr_sims) in enumerate(zip(indices, similarity_scores)):
    this_item_id = item_ids[item_col_idx]
    neighbors = []
    for nbr_col_idx, sim in zip(nbr_cols, nbr_sims):
        if nbr_col_idx == item_col_idx:
            continue
        neighbors.append((item_ids[nbr_col_idx], float(sim)))
    item_id_to_item_neighbors[this_item_id] = neighbors

# write the dict to disk as a joblib so you can reuse without recomputing if desired
joblib.dump(item_id_to_item_neighbors, NEIGHBORS_QUAL_PATH)
print(f"saved qualified neighbors to: {NEIGHBORS_QUAL_PATH}")

# display example neighbors
print("\n-- example of some neighbors for first 3 items --")
for i, (item_id, nbrs) in enumerate(item_id_to_item_neighbors.items()):
    if i >= 3:
        break
    print(f"\nitem {item_id}:")
    for n_id, s in nbrs[:10]:
        print(f"  -> {n_id}: sim={s:.4f}")

# -------------------------------------------------------------------
# 5) Build user -> items history lookup from strength_df
# -------------------------------------------------------------------

print_section_header("5) Building user -> items lookup")

# lookup for a user to all of their interacted content
# of the form {adventurer_id: [(content_id_a, strength_a), (content_id_b, strength_b), ...], adventurer_id: ...}
user_hist = (
    strength_df.groupby("adventurer_id")[["content_id", "strength"]]
    .apply(lambda x: list(zip(x["content_id"], x["strength"])))
    .to_dict()
)

print(f"user -> items-history dict built for {len(user_hist):,} users")

# -------------------------------------------------------------------
# 6) Discover users to score (all + warm test)
# -------------------------------------------------------------------

print_section_header("6) Discovering warm users to score for recommendations")

train_users_set = set(user_ids)  # all user ids in qualified training set

# get all distinct users from test parquet
test_users = con.execute(f"""
    select distinct adventurer_id
    from read_parquet('{TEST_SET_PATH}')
""").df()["adventurer_id"].tolist()
test_users_set = set(test_users)

# “warm” here means they appear in both the train and test sets (so the model can score them).
# the item–item CF model here needs history, so cold-start users are excluded.
# due to temporal split, if you appear in the test but not the train then you are a cold-start user
# and conversly if you appear in the train only then you have no testing value.
# so "warm" users only for this model and testing purposes.
warm_test_users = sorted(train_users_set.intersection(test_users_set))

print(f"qualified train users                            = {len(train_users_set):,}")
print(f"raw test users                                   = {len(test_users_set):,}")
print(f"\"warm\" test users (are in both train and test)   = {len(warm_test_users):,}")

# -------------------------------------------------------------------
# 7) Score users and save the recommendations
# -------------------------------------------------------------------

print_section_header(f"7) Doing recommendation scoring for users")

# core helper for recommendation scoring of a single user
def score_single_user_top_k(user_id, topk=10):

    history = user_hist.get(user_id, [])

    if not history:
        return []  # if no qualified content interactions then no recommendations

    seen_items = {item_id for item_id, _ in history}
    scores = defaultdict(float)

    for hist_item_id, hist_strength in history:
        nbrs = item_id_to_item_neighbors.get(hist_item_id, [])
        for nbr_item_id, sim in nbrs:
            if nbr_item_id in seen_items:
                continue
            scores[nbr_item_id] += sim * hist_strength

    if not scores:
        return []

    top_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topk]
    return [(item_id, float(score)) for item_id, score in top_items]

# loops users, gets their top-k recs, expands into one row per recommendation
# includes rank 1..K recommendations for each user
# does progress logging every 5000 users
# returns a dataframe with the schema: adventurer_id, rec_rank, content_id, score
def score_user_list(users, label):
    print(f" SCORING USERS: {label}", '\n')
    rows = []
    for i, u in enumerate(users):
        top_recs = score_single_user_top_k(u, topk=TOP_K_RECS)
        for rank, (item_id, score) in enumerate(top_recs, start=1):
            rows.append({
                "adventurer_id": u,
                "rec_rank": rank,
                "content_id": item_id,
                "score": score
            })
        if (i + 1) % 5000 == 0:
            print(f"scored {i + 1:,} users...")
    rec_df = pd.DataFrame(rows)
    print(f"produced {len(rec_df):,} recommendation rows")
    return rec_df

# get recommendations for all qualified-train users
recs_all_df = score_user_list(sorted(train_users_set), "ALL QUALIFIED TRAIN USERS")
con.register("recs_all_df", recs_all_df)
con.execute(f"copy recs_all_df to '{RECS_ALL_QUAL_PATH}' (format parquet)")
print(f"saved all-user recs to: {RECS_ALL_QUAL_PATH}", '\n')

# get recommendations for warm test users
recs_test_df = score_user_list(warm_test_users, "WARM TEST USERS (QUALIFIED MODEL)")
con.register("recs_test_df", recs_test_df)
con.execute(f"copy recs_test_df to '{RECS_TEST_QUAL_PATH}' (format parquet)")
print(f"saved test-user recs to: {RECS_TEST_QUAL_PATH}")

# -------------------------------------------------------------------
# 8) Sample recommendations
# -------------------------------------------------------------------

print_section_header("8) Sample recommendations")

print("qualified all-train-users sample...")
print(recs_all_df.head(20).to_string(index=False))
print("\nqualified warm test-users sample...")
print(recs_test_df.head(20).to_string(index=False))

# print("\n" + "=" * 95)
# print("DONE.")
# print("=" * 95)
print_section_header("DONE.", no_endline=True)

con.close()