"""Fetch Strava activities via the official API and cache them locally.

The cache lives in data/activities.json. Syncs are incremental: only
activities newer than the most recent cached one are downloaded, which keeps
us comfortably inside Strava's rate limits (100 requests / 15 min by default).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

from strava_auth import get_valid_access_token

API_BASE = "https://www.strava.com/api/v3"
DATA_DIR = Path(__file__).with_name("data")
ACTIVITIES_PATH = DATA_DIR / "activities.json"

# Summary-activity fields we keep. Anything missing for a given activity
# (e.g. heart rate on a walk) is filled with NA so the schema stays stable.
KEEP_COLUMNS = [
    "id",
    "name",
    "type",
    "sport_type",
    "start_date",          # UTC, used for incremental sync
    "start_date_local",    # local time, used for analysis
    "distance",            # metres
    "moving_time",         # seconds
    "elapsed_time",        # seconds
    "total_elevation_gain",  # metres
    "average_speed",       # m/s
    "max_speed",           # m/s
    "average_heartrate",   # bpm
    "max_heartrate",       # bpm
    "average_cadence",
    "average_watts",
    "kilojoules",
    "suffer_score",        # Strava "relative effort"
    "achievement_count",
    "kudos_count",
    "elev_high",
    "elev_low",
    "summary_polyline",    # encoded route shape (from the nested "map" object)
]

BEST_EFFORTS_PATH = DATA_DIR / "best_efforts.json"

# Activity types that have running "best efforts" (fastest 1k/5k/10k/...).
RUN_TYPES = {"Run", "TrailRun", "VirtualRun"}


class RateLimitError(RuntimeError):
    pass


def _get(path: str, token: str, params: dict | None = None):
    resp = requests.get(
        f"{API_BASE}{path}",
        headers={"Authorization": f"Bearer {token}"},
        params=params,
        timeout=30,
    )
    if resp.status_code == 429:
        raise RateLimitError(
            "Strava rate limit reached. Wait ~15 minutes and sync again."
        )
    resp.raise_for_status()
    return resp.json()


def fetch_activities(after_epoch: int | None = None) -> list[dict]:
    """Download summary activities, newest first paginated, optionally only
    those started after ``after_epoch`` (unix seconds, UTC)."""
    token = get_valid_access_token()
    out: list[dict] = []
    page = 1
    while True:
        params = {"per_page": 200, "page": page}
        if after_epoch:
            params["after"] = after_epoch
        batch = _get("/athlete/activities", token, params)
        if not batch:
            break
        out.extend(batch)
        page += 1
        time.sleep(0.2)  # be gentle with the API
    return out


def _records_to_frame(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    # Pull the route shape out of the nested "map" object before trimming columns.
    if "map" in df.columns:
        df["summary_polyline"] = df["map"].apply(
            lambda m: m.get("summary_polyline") if isinstance(m, dict) else None
        )
    for col in KEEP_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[KEEP_COLUMNS].copy()
    df["start_date"] = pd.to_datetime(df["start_date"], utc=True)
    df["start_date_local"] = pd.to_datetime(df["start_date_local"]).dt.tz_localize(None)
    return df


def load_cached() -> pd.DataFrame | None:
    """Load the cached activities as a DataFrame, or None if nothing cached."""
    if not ACTIVITIES_PATH.exists():
        return None
    records = json.loads(ACTIVITIES_PATH.read_text())
    if not records:
        return None
    return _records_to_frame(records)


def _save(df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    # Store as ISO strings so the JSON cache is portable and human-readable.
    out = df.copy()
    out["start_date"] = out["start_date"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out["start_date_local"] = out["start_date_local"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    ACTIVITIES_PATH.write_text(out.to_json(orient="records", indent=2))


def sync(force_full: bool = False) -> pd.DataFrame:
    """Fetch new activities and merge them into the local cache.

    With ``force_full=True`` the whole history is re-downloaded.
    Returns the full, de-duplicated, chronologically sorted DataFrame.
    """
    existing = None if force_full else load_cached()

    after = None
    if existing is not None and not existing.empty:
        after = int(existing["start_date"].max().timestamp())

    new_records = fetch_activities(after_epoch=after)

    if not new_records:
        return existing if existing is not None else pd.DataFrame(columns=KEEP_COLUMNS)

    new_df = _records_to_frame(new_records)
    if existing is not None:
        df = pd.concat([existing, new_df], ignore_index=True)
    else:
        df = new_df

    df = (
        df.drop_duplicates(subset="id", keep="last")
        .sort_values("start_date_local")
        .reset_index(drop=True)
    )
    _save(df)
    return df


# --------------------------------------------------------------------------- #
# Best efforts (per-run fastest splits) — needs one detail call per run.
# --------------------------------------------------------------------------- #
# Stay safely under Strava's default 100 requests / 15 minutes.
_WINDOW_LIMIT = 90
_WINDOW_SLEEP = 15 * 60


def fetch_activity_detail(activity_id: int, token: str) -> dict:
    return _get(f"/activities/{activity_id}", token, {"include_all_efforts": True})


def _load_best_efforts_raw() -> dict:
    if BEST_EFFORTS_PATH.exists():
        data = json.loads(BEST_EFFORTS_PATH.read_text())
        data.setdefault("checked_ids", [])
        data.setdefault("efforts", [])
        return data
    return {"checked_ids": [], "efforts": []}


def _save_best_efforts(data: dict) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    BEST_EFFORTS_PATH.write_text(json.dumps(data, indent=2))


def load_best_efforts() -> pd.DataFrame | None:
    """Best efforts as one row per effort, or None if nothing cached yet."""
    data = _load_best_efforts_raw()
    if not data["efforts"]:
        return None
    df = pd.DataFrame(data["efforts"])
    df["start_date_local"] = pd.to_datetime(df["start_date_local"])
    return df


def best_efforts_progress() -> tuple[int, int]:
    """(runs_checked, total_runs) — for progress display."""
    activities = load_cached()
    if activities is None:
        return (0, 0)
    total = int(activities["sport_type"].isin(RUN_TYPES).sum())
    checked = len(set(_load_best_efforts_raw()["checked_ids"]))
    return (checked, total)


def sync_best_efforts(max_requests: int | None = None, sleep_on_limit: bool = True,
                      log=print) -> pd.DataFrame | None:
    """Fetch per-run best efforts for runs not yet checked.

    Resumable: every run id is recorded once checked, so re-runs continue where
    they left off. Stays under the rate limit; when a 15-minute window is used
    up it either pauses (``sleep_on_limit=True``, for an unattended full pull)
    or stops (used by the dashboard button with a bounded ``max_requests``).
    """
    activities = load_cached()
    if activities is None:
        return None

    data = _load_best_efforts_raw()
    checked = set(data["checked_ids"])

    runs = activities[activities["sport_type"].isin(RUN_TYPES)].sort_values(
        "start_date_local", ascending=False
    )
    to_fetch = [
        (int(row["id"]), row["name"], str(row["start_date_local"]))
        for _, row in runs[["id", "name", "start_date_local"]].iterrows()
        if int(row["id"]) not in checked
    ]
    if not to_fetch:
        log("Best efforts already up to date.")
        return load_best_efforts()

    token = get_valid_access_token()
    calls = window_calls = 0
    for activity_id, activity_name, start_local in to_fetch:
        if max_requests is not None and calls >= max_requests:
            break
        if window_calls >= _WINDOW_LIMIT:
            _save_best_efforts(data)
            if not sleep_on_limit:
                break
            log(f"Rate-limit window used after {calls} fetched; sleeping 15 min…")
            time.sleep(_WINDOW_SLEEP)
            window_calls = 0
            token = get_valid_access_token()

        try:
            detail = fetch_activity_detail(activity_id, token)
        except RateLimitError:
            _save_best_efforts(data)
            if not sleep_on_limit:
                break
            log("Rate limit hit; sleeping 15 min…")
            time.sleep(_WINDOW_SLEEP)
            window_calls = 0
            token = get_valid_access_token()
            continue  # retry the same activity

        calls += 1
        window_calls += 1
        for eff in detail.get("best_efforts") or []:
            data["efforts"].append({
                "activity_id": activity_id,
                "activity_name": activity_name,
                "start_date_local": start_local,
                "name": eff.get("name"),
                "distance": eff.get("distance"),
                "elapsed_time": eff.get("elapsed_time"),
                "moving_time": eff.get("moving_time"),
            })
        data["checked_ids"].append(activity_id)

        if calls % 25 == 0:
            _save_best_efforts(data)
            log(f"Fetched {calls} / {len(to_fetch)} runs…")
        time.sleep(0.2)

    _save_best_efforts(data)
    log(f"Done: fetched {calls} this run; {len(data['checked_ids'])} of "
        f"{len(runs)} runs checked; {len(data['efforts'])} efforts recorded.")
    return load_best_efforts()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "best":
        sync_best_efforts()
    else:
        frame = sync()
        print(f"Cached {len(frame)} activities to {ACTIVITIES_PATH}")
