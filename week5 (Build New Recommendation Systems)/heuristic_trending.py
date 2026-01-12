# heuristic model
# this is a simple heuristic recommender based on "trending" items.
# still uses watch_ratio >= QUALIFIED_THRESHOLD for qualifying a view
# scores items globally using recency-weighted qualified views (exponential decay by event_day_index)
# recommends the top-K trending items per user, excluding items the user has already seen in TRAIN
# results though in nearly identical recommendations for everyone of course
#
# good for cold-start test users (users not present in train)
# for warm users, we exclude items they already saw in the train split (all views, not just qualified ones)

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path

# helper for readable console logs
def print_section_header(title, no_endline=False):
    print("\n" + "=" * 95)
    print(title)
    print("=" * 95)
    if not no_endline:
        print()

# CONFIG VARIABLES

QUALIFIED_THRESHOLD = 0.60
TOP_K_RECS = 10

HALF_LIFE_DAYS = 48

CANDIDATE_POOL_SIZE = 5000

TRAIN_SET_PATH = "data/data-parquet-files/test-train-splits/content_views_train_split.parquet"
TEST_SET_PATH = "data/data-parquet-files/test-train-splits/content_views_test_split.parquet"

OUT_DIR = "data/recommender-output-files"

# outputs
TRAIN_QUALIFIED_PATH = f"{OUT_DIR}/content_views_train_split_qualified_only.parquet"
TRENDING_ITEM_SCORES_PATH = f"{OUT_DIR}/trending_item_scores_halfLife{HALF_LIFE_DAYS}.parquet"
RECS_TREND_ALL_TEST_PATH = f"{OUT_DIR}/recs_trending_all_test_users_top{TOP_K_RECS}.parquet"
RECS_TREND_WARM_TEST_PATH = f"{OUT_DIR}/recs_trending_warm_test_users_top{TOP_K_RECS}.parquet"

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# Create "qualified"-only TRAIN parquet, same as in the CF pipeline

con = duckdb.connect()

print_section_header('1) Writing "qualified"-only TRAIN parquet file')

con.execute(f"""
    copy (
        select *
        from read_parquet('{TRAIN_SET_PATH}')
        where watch_ratio >= {QUALIFIED_THRESHOLD}
    )
    to '{TRAIN_QUALIFIED_PATH}' (format parquet)
""")

qual_train_counts = con.execute(f"""
    select
        count(*) as qualified_train_rows,
        count(distinct adventurer_id) as qualified_train_users,
        count(distinct content_id) as qualified_train_items,
        min(event_day_index) as min_event_day_index,
        max(event_day_index) as max_event_day_index
    from read_parquet('{TRAIN_QUALIFIED_PATH}')
""").df()

print(f"qualified threshold used is watch_ratio >= {QUALIFIED_THRESHOLD}\n")
print(f"wrote qualified TRAIN to: {TRAIN_QUALIFIED_PATH}")
print(qual_train_counts.to_string(index=False))

# Compute trending scores per item from qualified train (recency-weighted)

print_section_header("2) Computing trending item scores from qualified train")

lambda_decay = float(np.log(2.0) / HALF_LIFE_DAYS)

# compute global trending score per item
# score(content) = sum(exp(-lambda * (Dmax - event_day_index))) over qualified views
item_scores_df = con.execute(f"""
    with q as (
        select
            content_id,
            event_day_index
        from read_parquet('{TRAIN_QUALIFIED_PATH}')
    ),
    stats as (
        select
            max(event_day_index) as dmax
        from q
    )
    select
        q.content_id,
        sum(exp(-{lambda_decay} * (stats.dmax - q.event_day_index))) as score,
        count(*) as qualified_view_count
    from q
    cross join stats
    group by q.content_id
    order by score desc
""").df()

print(f"computed scores for {len(item_scores_df):,} items")
print("\nscore sample (top 15):")
print(item_scores_df.head(15).to_string(index=False))

# saving scores for inspection
con.register("item_scores_df", item_scores_df)
con.execute(f"copy item_scores_df to '{TRENDING_ITEM_SCORES_PATH}' (format parquet)")
print(f"\nsaved trending item scores to: {TRENDING_ITEM_SCORES_PATH}")

# build a ranked candidate list
ranked_items = item_scores_df["content_id"].tolist()
ranked_scores = item_scores_df["score"].astype(float).tolist()

if CANDIDATE_POOL_SIZE is not None and CANDIDATE_POOL_SIZE > 0:
    ranked_items = ranked_items[: min(CANDIDATE_POOL_SIZE, len(ranked_items))]
    ranked_scores = ranked_scores[: min(CANDIDATE_POOL_SIZE, len(ranked_scores))]

# dictionary for quick score lookup when creating rec rows
item_score_map = {cid: float(s) for cid, s in zip(ranked_items, ranked_scores)}

print(f"\nusing candidate pool size = {len(ranked_items):,} (top trending items)")

# Discover users to score (all test users + warm test users)

print_section_header("3) Discovering test users and warm test users (train intersect test)")

train_users = con.execute(f"""
    select distinct adventurer_id
    from read_parquet('{TRAIN_SET_PATH}')
""").df()["adventurer_id"].tolist()
train_users_set = set(train_users)

test_users = con.execute(f"""
    select distinct adventurer_id
    from read_parquet('{TEST_SET_PATH}')
""").df()["adventurer_id"].tolist()
test_users_set = set(test_users)

warm_test_users = sorted(train_users_set.intersection(test_users_set))

print(f"train users                                 = {len(train_users_set):,}")
print(f"test users                                  = {len(test_users_set):,}")
print(f"warm test users (are in both train and test) = {len(warm_test_users):,}")


# Build warm user

print_section_header("4) Building warm user")

warm_seen_df = con.execute(f"""
    select
        adventurer_id,
        list(distinct content_id) as seen_items
    from read_parquet('{TRAIN_SET_PATH}')
    where adventurer_id in (
        select unnest(?)
    )
    group by adventurer_id
""", [warm_test_users]).df()

warm_user_to_seen = dict(zip(warm_seen_df["adventurer_id"], warm_seen_df["seen_items"]))

print(f"built seen-items lists for {len(warm_user_to_seen):,} warm users")
print("\nsample warm user seen-items sizes:")
sample_sizes = warm_seen_df.head(10).copy()
sample_sizes["seen_count"] = sample_sizes["seen_items"].apply(lambda x: 0 if x is None else len(x))
print(sample_sizes[["adventurer_id", "seen_count"]].to_string(index=False))

# Recommend: for each user, take top-K trending items not in seen set

print_section_header(f"5) Scoring and saving recommendations (TOP_K_RECS = {TOP_K_RECS})")

def top_k_unseen_for_user(user_id, topk):
    # warm users have a seen list; cold users default to empty set
    seen_list = warm_user_to_seen.get(user_id, None)
    seen_set = set(seen_list) if seen_list is not None else set()

    recs = []
    for cid in ranked_items:
        if cid in seen_set:
            continue
        recs.append((cid, item_score_map.get(cid, 0.0)))
        if len(recs) >= topk:
            break
    return recs

def score_user_list(users, label):
    print(f" SCORING USERS: {label}\n")
    rows = []
    for i, u in enumerate(users):
        top_recs = top_k_unseen_for_user(u, topk=TOP_K_RECS)
        for rank, (cid, score) in enumerate(top_recs, start=1):
            rows.append({
                "adventurer_id": u,
                "rec_rank": rank,
                "content_id": cid,
                "score": float(score),
            })
        if (i + 1) % 5000 == 0:
            print(f"scored {i + 1:,} users...")

    rec_df = pd.DataFrame(rows)
    print(f"produced {len(rec_df):,} recommendation rows")
    return rec_df

# all test users (includes cold-start users)
recs_all_test_df = score_user_list(sorted(test_users_set), "ALL TEST USERS (TRENDING HEURISTIC)")
con.register("recs_all_test_df", recs_all_test_df)
con.execute(f"copy recs_all_test_df to '{RECS_TREND_ALL_TEST_PATH}' (format parquet)")
print(f"saved all-test-user recs to: {RECS_TREND_ALL_TEST_PATH}\n")

# warm test users only (apples-to-apples comparison with CF)
recs_warm_test_df = score_user_list(warm_test_users, "WARM TEST USERS (TRENDING HEURISTIC)")
con.register("recs_warm_test_df", recs_warm_test_df)
con.execute(f"copy recs_warm_test_df to '{RECS_TREND_WARM_TEST_PATH}' (format parquet)")
print(f"saved warm-test-user recs to: {RECS_TREND_WARM_TEST_PATH}")

# Sample output

print_section_header("6) Sample recommendations")

print("trending all-test-users sample...")
print(recs_all_test_df.head(20).to_string(index=False))
print("\ntrending warm-test-users sample...")
print(recs_warm_test_df.head(20).to_string(index=False))

print_section_header("DONE.", no_endline=True)
con.close()