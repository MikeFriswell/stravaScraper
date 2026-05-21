"""Turn raw Strava activities into analysis-ready data and metrics.

Strava stores SI units: distance/elevation in metres, durations in seconds,
speeds in metres per second. Everything user-facing is derived here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

M_PER_KM = 1000.0
M_PER_MILE = 1609.344
M_PER_FOOT = 0.3048

# Sport types Strava reports a pace for (vs. speed for ride/swim, etc.).
FOOT_SPORTS = {"Run", "TrailRun", "VirtualRun", "Walk", "Hike"}

# Pure running types — used as the default activity filter so walks/rides don't
# skew the pace and distance axes.
RUN_SPORTS = {"Run", "TrailRun", "VirtualRun"}


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived distance/pace/time/calendar columns to the activity frame."""
    df = df.copy()
    if df.empty:
        return df

    # Distance / elevation in friendly units.
    df["distance_km"] = df["distance"] / M_PER_KM
    df["distance_mi"] = df["distance"] / M_PER_MILE
    df["elevation_m"] = df["total_elevation_gain"]
    df["elevation_ft"] = df["total_elevation_gain"] / M_PER_FOOT

    # Durations.
    df["moving_min"] = df["moving_time"] / 60.0
    df["moving_hr"] = df["moving_time"] / 3600.0

    # Speed and pace. Guard against zero/NaN speeds.
    spd = df["average_speed"].replace(0, np.nan)
    df["speed_kmh"] = spd * 3.6
    df["speed_mph"] = spd * 2.2369362921
    df["pace_min_per_km"] = (M_PER_KM / spd) / 60.0
    df["pace_min_per_mi"] = (M_PER_MILE / spd) / 60.0

    # Aerobic efficiency: metres travelled per minute per heartbeat.
    # Higher = more speed for the same heart rate (a fitness proxy).
    hr = df["average_heartrate"].replace(0, np.nan)
    df["aerobic_ef"] = (spd * 60.0) / hr

    # Calendar features from local start time.
    local = df["start_date_local"]
    df["date"] = local.dt.date
    df["year"] = local.dt.year
    df["month"] = local.dt.to_period("M").dt.to_timestamp()
    df["week"] = local.dt.to_period("W").dt.start_time
    df["day_of_week"] = local.dt.day_name()
    df["dow_num"] = local.dt.dayofweek
    df["hour"] = local.dt.hour

    return df


def unit_columns(metric: bool) -> dict:
    """Map logical metrics to the right column + label for the chosen unit system."""
    if metric:
        return {
            "distance": ("distance_km", "Distance (km)", "km"),
            "elevation": ("elevation_m", "Elevation (m)", "m"),
            "speed": ("speed_kmh", "Speed (km/h)", "km/h"),
            "pace": ("pace_min_per_km", "Pace (min/km)", "min/km"),
        }
    return {
        "distance": ("distance_mi", "Distance (mi)", "mi"),
        "elevation": ("elevation_ft", "Elevation (ft)", "ft"),
        "speed": ("speed_mph", "Speed (mph)", "mph"),
        "pace": ("pace_min_per_mi", "Pace (min/mi)", "min/mi"),
    }


def format_pace(minutes: float) -> str:
    """Format decimal minutes as m:ss (e.g. 4.5 -> '4:30'). NaN -> '—'."""
    if minutes is None or pd.isna(minutes) or not np.isfinite(minutes):
        return "—"
    total_seconds = int(round(minutes * 60))
    return f"{total_seconds // 60}:{total_seconds % 60:02d}"


def format_duration(seconds: float) -> str:
    """Format seconds as H:MM:SS (or M:SS under an hour)."""
    if seconds is None or pd.isna(seconds):
        return "—"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def period_summary(df: pd.DataFrame, period: str, dist_col: str) -> pd.DataFrame:
    """Aggregate volume by 'week', 'month', or 'year'."""
    key = {"week": "week", "month": "month", "year": "year"}[period]
    grouped = (
        df.groupby(key)
        .agg(
            activities=("id", "count"),
            distance=(dist_col, "sum"),
            moving_hr=("moving_hr", "sum"),
            elevation=("elevation_m", "sum"),
        )
        .reset_index()
        .rename(columns={key: "period"})
    )
    return grouped


# --------------------------------------------------------------------------- #
# Fitness & Form (training-load model)
# --------------------------------------------------------------------------- #
def _activity_load(df: pd.DataFrame) -> pd.Series:
    """Per-activity training load: relative effort where available, otherwise
    estimated from moving time using the median effort-per-minute."""
    effort = df["suffer_score"]
    has = effort.notna() & (df["moving_min"] > 0)
    per_min = float((effort[has] / df.loc[has, "moving_min"]).median()) if has.any() else 1.0
    if not np.isfinite(per_min) or per_min <= 0:
        per_min = 1.0
    return effort.fillna(df["moving_min"] * per_min)


def training_load(df: pd.DataFrame, tc_fitness: int = 42, tc_fatigue: int = 7) -> pd.DataFrame:
    """Daily Fitness (CTL), Fatigue (ATL) and Form (TSB) from training load.

    Fitness/Fatigue are exponentially weighted averages of daily load over
    ~42 / ~7 day windows; Form is yesterday's (Fitness − Fatigue).
    """
    cols = ["date", "load", "fitness", "fatigue", "form"]
    if df.empty:
        return pd.DataFrame(columns=cols)

    load = _activity_load(df)
    daily = (
        df.assign(load=load)
        .groupby(df["start_date_local"].dt.normalize())["load"]
        .sum()
    )
    full = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full, fill_value=0.0)

    fitness = daily.ewm(alpha=1 / tc_fitness, adjust=False).mean()
    fatigue = daily.ewm(alpha=1 / tc_fatigue, adjust=False).mean()
    form = (fitness - fatigue).shift(1)
    return pd.DataFrame({
        "date": daily.index, "load": daily.values,
        "fitness": fitness.values, "fatigue": fatigue.values, "form": form.values,
    })


# --------------------------------------------------------------------------- #
# Consistency: streaks, calendar, workload ratio
# --------------------------------------------------------------------------- #
def _active_days(df: pd.DataFrame) -> pd.Series:
    return pd.Series(sorted(df["start_date_local"].dt.normalize().unique()))


def streaks(df: pd.DataFrame) -> dict:
    """Longest/current consecutive-day streaks plus rest-day stats."""
    days = _active_days(df)
    if days.empty:
        return {"longest_streak": 0, "current_streak": 0,
                "total_active_days": 0, "longest_gap": 0}
    gaps = days.diff().dt.days
    groups = (gaps != 1).cumsum()
    sizes = days.groupby(groups).size()
    last = days.iloc[-1]
    days_since = (pd.Timestamp.today().normalize() - last).days
    return {
        "longest_streak": int(sizes.max()),
        "current_streak": int(sizes.iloc[-1]) if days_since <= 1 else 0,
        "total_active_days": int(len(days)),
        "longest_gap": int(gaps.max()) if gaps.notna().any() else 0,
        "days_since_last": int(days_since),
    }


def weekly_consistency(df: pd.DataFrame, target: int = 3, today: pd.Timestamp | None = None):
    """Runs per week vs a weekly target (default 3), with a streak summary.

    Consistency is measured by weeks, not days: a week is 'on target' at
    ``target`` or more runs. Weeks below target still count toward your average
    (partial credit) — they just don't extend the on-target streak. The current
    in-progress week is shown but excluded from streak/percentage stats so an
    unfinished week never looks like a miss.
    """
    cols = ["week", "runs", "met", "status", "completed"]
    if df.empty:
        return pd.DataFrame(columns=cols), {}

    today = (today or pd.Timestamp.today()).normalize()
    cur_week = today.to_period("W").start_time

    counts = df.groupby("week").size()
    weeks = pd.date_range(counts.index.min(), max(counts.index.max(), cur_week), freq="W-MON")
    counts = counts.reindex(weeks, fill_value=0)

    frame = pd.DataFrame({"week": counts.index, "runs": counts.to_numpy(dtype=int)})
    frame["completed"] = (today - frame["week"]).dt.days >= 7
    frame["met"] = frame["runs"] >= target

    def _status(row):
        if row["week"] == cur_week and not row["completed"]:
            return "This week"
        if row["runs"] >= target:
            return "Target met"
        return "Partial" if row["runs"] > 0 else "Rest week"

    frame["status"] = frame.apply(_status, axis=1)

    comp = frame[frame["completed"]]
    met_seq = comp["met"].tolist()

    current = 0
    for met in reversed(met_seq):
        if met:
            current += 1
        else:
            break
    longest = run = 0
    for met in met_seq:
        run = run + 1 if met else 0
        longest = max(longest, run)
    active = 0
    for runs in reversed(comp["runs"].tolist()):
        if runs >= 1:
            active += 1
        else:
            break

    stats = {
        "target": target,
        "avg_per_week": float(comp["runs"].mean()) if len(comp) else 0.0,
        "weeks_total": int(len(comp)),
        "weeks_met": int(comp["met"].sum()),
        "pct_met": float(comp["met"].mean() * 100) if len(comp) else 0.0,
        "current_streak": current,
        "longest_streak": longest,
        "active_week_streak": active,
    }
    return frame, stats


def calendar_matrix(df: pd.DataFrame, year: int, dist_col: str):
    """7×(weeks) matrix of daily distance for a GitHub-style calendar heatmap.

    Rows are weekday (Mon..Sun), columns are week-of-year. Returns the matrix
    and a per-month (label, column) list for x-axis ticks.
    """
    start, end = pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12, 31)
    idx = pd.date_range(start, end, freq="D")
    in_year = df[(df["start_date_local"] >= start) &
                 (df["start_date_local"] < end + pd.Timedelta(days=1))]
    daily = (
        in_year.groupby(in_year["start_date_local"].dt.normalize())[dist_col]
        .sum().reindex(idx, fill_value=0.0)
    )
    frame = pd.DataFrame({"date": idx, "dist": daily.values})
    frame["weekday"] = frame["date"].dt.weekday
    frame["week_idx"] = ((frame["date"] - start).dt.days + start.weekday()) // 7
    matrix = (
        frame.pivot_table(index="weekday", columns="week_idx", values="dist",
                          aggfunc="sum", fill_value=0.0)
        .reindex(range(7), fill_value=0.0)
    )
    month_ticks = [
        (frame.loc[frame["date"].dt.month == m, "week_idx"].iloc[0],
         pd.Timestamp(year, m, 1).strftime("%b"))
        for m in range(1, 13)
    ]
    return matrix, month_ticks


def workload_ratio(df: pd.DataFrame, dist_col: str) -> pd.DataFrame:
    """Acute (7-day) vs chronic (28-day) average daily load and their ratio.

    An ACWR roughly in 0.8–1.3 is the commonly cited 'sweet spot'; spikes
    above ~1.5 flag rapid load increases associated with injury risk.
    """
    if df.empty:
        return pd.DataFrame(columns=["date", "acute", "chronic", "acwr"])
    daily = (
        df.groupby(df["start_date_local"].dt.normalize())[dist_col]
        .sum()
    )
    full = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full, fill_value=0.0)
    acute = daily.rolling(7).mean()
    chronic = daily.rolling(28).mean()
    acwr = (acute / chronic).replace([np.inf, -np.inf], np.nan)
    return pd.DataFrame({"date": daily.index, "acute": acute.values,
                         "chronic": chronic.values, "acwr": acwr.values})


# --------------------------------------------------------------------------- #
# Year-over-year
# --------------------------------------------------------------------------- #
def yearly_cumulative(df: pd.DataFrame, dist_col: str) -> pd.DataFrame:
    """Cumulative distance by day-of-year, one series per calendar year."""
    if df.empty:
        return pd.DataFrame(columns=["year", "doy", "cumulative"])
    d = df.assign(
        doy=df["start_date_local"].dt.dayofyear,
        yr=df["start_date_local"].dt.year,
    )
    daily = d.groupby(["yr", "doy"])[dist_col].sum().reset_index()
    daily = daily.sort_values(["yr", "doy"])
    daily["cumulative"] = daily.groupby("yr")[dist_col].cumsum()
    return daily.rename(columns={"yr": "year"})


# --------------------------------------------------------------------------- #
# Best efforts / race prediction
# --------------------------------------------------------------------------- #
# Strava's best-effort labels, shortest to longest.
BEST_EFFORT_ORDER = [
    "400m", "1/2 mile", "1K", "1 mile", "2 mile", "5K", "10K",
    "15K", "10 mile", "20K", "Half-Marathon", "Marathon",
]

RACE_DISTANCES = {  # metres
    "1 mile": 1609.344, "5K": 5000.0, "10K": 10000.0,
    "Half-Marathon": 21097.5, "Marathon": 42195.0,
}


def best_effort_prs(efforts: pd.DataFrame | None) -> pd.DataFrame:
    """Fastest (min elapsed time) effort for each distance label."""
    if efforts is None or efforts.empty:
        return pd.DataFrame()
    idx = efforts.groupby("name")["elapsed_time"].idxmin()
    pr = efforts.loc[idx].copy()
    order = {n: i for i, n in enumerate(BEST_EFFORT_ORDER)}
    pr["order"] = pr["name"].map(order).fillna(99)
    pr["pace_min_per_km"] = (pr["distance"] / 1000.0).rdiv(pr["elapsed_time"] / 60.0)
    return pr.sort_values("order").drop(columns="order")


def riegel_predict(base_distance_m: float, base_time_s: float,
                   targets: dict | None = None, exponent: float = 1.06) -> dict:
    """Predict race times from one performance via Riegel's formula:
    T2 = T1 * (D2 / D1) ** 1.06."""
    targets = targets or RACE_DISTANCES
    return {name: base_time_s * (dist / base_distance_m) ** exponent
            for name, dist in targets.items()}


def fit_fatigue_exponent(
    efforts: pd.DataFrame | None,
    window_days: int | None = 365,
    min_distance_m: float = 1000.0,
) -> tuple[float, float, int] | None:
    """Fit your personal fatigue exponent b in T = a · D**b by least-squares on
    log(time) vs log(distance) over your best efforts (≥ ``min_distance_m``).

    Returns (exponent, r_squared, n_distances), or None if fewer than two
    distinct distances are available to fit. b ≈ 1.06 is Riegel's population
    average; higher means you slow more as distance grows.
    """
    if efforts is None or efforts.empty:
        return None
    e = efforts.copy()
    if window_days is not None:
        asof = e["start_date_local"].max()
        e = e[e["start_date_local"] >= asof - pd.Timedelta(days=window_days)]
    bests = e.loc[e.groupby("name")["elapsed_time"].idxmin(), ["distance", "elapsed_time"]]
    bests = bests[(bests["distance"] >= min_distance_m) & (bests["elapsed_time"] > 0)]
    if bests["distance"].nunique() < 2:
        return None

    x = np.log(bests["distance"].to_numpy(dtype=float))
    y = np.log(bests["elapsed_time"].to_numpy(dtype=float))
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (intercept + slope * x)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), r2, int(len(bests))


def predict_with_uncertainty(
    efforts: pd.DataFrame | None,
    window_days: int | None = 365,
    exponent: float = 1.06,
    targets: dict | None = None,
) -> pd.DataFrame:
    """Ensemble race-time predictions with an uncertainty spread.

    For each target distance, every recorded best effort is extrapolated with
    Riegel's formula, then combined with weights that favour efforts at nearby
    distances (where Riegel is most accurate). The weighted mean is the
    prediction; the weighted standard deviation is the spread (how much your
    efforts disagree). Only efforts within ``window_days`` of your most recent
    effort are used so stale performances don't dominate — ``None`` = all time.
    """
    targets = targets or RACE_DISTANCES
    cols = ["race", "distance", "predicted_s", "std_s", "low_s", "high_s", "n_efforts"]
    if efforts is None or efforts.empty:
        return pd.DataFrame(columns=cols)

    e = efforts.copy()
    if window_days is not None:
        asof = e["start_date_local"].max()
        e = e[e["start_date_local"] >= asof - pd.Timedelta(days=window_days)]
    if e.empty:
        return pd.DataFrame(columns=cols)

    # Fastest effort per distance label within the window.
    bests = e.loc[e.groupby("name")["elapsed_time"].idxmin(), ["distance", "elapsed_time"]]
    bests = bests[(bests["distance"] > 0) & (bests["elapsed_time"] > 0)]
    if bests.empty:
        return pd.DataFrame(columns=cols)

    d1 = bests["distance"].to_numpy(dtype=float)
    t1 = bests["elapsed_time"].to_numpy(dtype=float)

    rows = []
    for race, d2 in targets.items():
        preds = t1 * (d2 / d1) ** exponent
        weights = 1.0 / (1.0 + np.abs(np.log(d2 / d1)))  # closer distance -> more weight
        wmean = float(np.average(preds, weights=weights))
        wstd = float(np.sqrt(np.average((preds - wmean) ** 2, weights=weights)))
        rows.append({
            "race": race, "distance": d2, "predicted_s": wmean, "std_s": wstd,
            "low_s": max(wmean - wstd, 0.0), "high_s": wmean + wstd,
            "n_efforts": int(len(preds)),
        })
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------- #
# Race planning
# --------------------------------------------------------------------------- #
# Target races. predict_key maps to a label in RACE_DISTANCES for predictions.
RACES = [
    {
        "name": "Two Castles 10K",
        "date": "2026-06-14",
        "distance_m": 10000.0,
        "predict_key": "10K",
        "where": "Warwick Castle → Kenilworth Castle",
        "notes": "Predominantly flat; main climb leaving Leek Wootton (~halfway). "
                 "June race — watch the heat.",
    },
    {
        "name": "The Big Half",
        "date": "2026-09-06",
        "distance_m": 21097.5,
        "predict_key": "Half-Marathon",
        "where": "Tower Bridge → Cutty Sark, Greenwich",
        "notes": "Fast and flat (~66 m gain).",
    },
]


def days_until(race_date: str, today: pd.Timestamp | None = None) -> int:
    today = (today or pd.Timestamp.today()).normalize()
    return int((pd.Timestamp(race_date).normalize() - today).days)


def goal_splits(distance_m: float, goal_seconds: float, unit_m: float):
    """Even-pace split plan: cumulative target time at each whole km/mile plus
    the finish. Returns (DataFrame[mark, cum_s], seconds_per_unit)."""
    total_units = distance_m / unit_m
    pace_per_unit_s = goal_seconds / total_units
    rows, k = [], 1
    while k < total_units - 1e-6:
        rows.append({"mark": float(k), "cum_s": pace_per_unit_s * k})
        k += 1
    rows.append({"mark": round(total_units, 2), "cum_s": float(goal_seconds)})
    return pd.DataFrame(rows), pace_per_unit_s


def weekly_longest_run(df: pd.DataFrame, dist_col: str) -> pd.DataFrame:
    """Longest single run per week (for endurance build-up tracking)."""
    if df.empty:
        return pd.DataFrame(columns=["week", "longest"])
    out = (
        df.groupby("week")[dist_col].max()
        .reset_index().rename(columns={dist_col: "longest"})
        .sort_values("week")
    )
    return out


# --------------------------------------------------------------------------- #
# Routes (encoded polyline decoding)
# --------------------------------------------------------------------------- #
def decode_polyline(encoded: str, precision: int = 5) -> list[tuple[float, float]]:
    """Decode a Google/Strava encoded polyline into [(lat, lon), ...]."""
    if not encoded:
        return []
    coords, index, lat, lng = [], 0, 0, 0
    factor = 10 ** precision
    length = len(encoded)
    while index < length:
        for is_lng in (False, True):
            shift = result = 0
            while True:
                b = ord(encoded[index]) - 63
                index += 1
                result |= (b & 0x1f) << shift
                shift += 5
                if b < 0x20:
                    break
            delta = ~(result >> 1) if (result & 1) else (result >> 1)
            if is_lng:
                lng += delta
            else:
                lat += delta
        coords.append((lat / factor, lng / factor))
    return coords


def route_points(df: pd.DataFrame, every: int = 3, max_points: int = 40000) -> pd.DataFrame:
    """All decoded route points (for a density heatmap), thinned and capped."""
    rows: list[tuple[float, float]] = []
    for poly in df["summary_polyline"].dropna():
        pts = decode_polyline(poly)
        rows.extend(pts[::every])
    if not rows:
        return pd.DataFrame(columns=["lat", "lon"])
    out = pd.DataFrame(rows, columns=["lat", "lon"])
    if len(out) > max_points:
        out = out.sample(max_points, random_state=0)
    return out


def route_lines(df: pd.DataFrame, every: int = 2, max_routes: int | None = None) -> pd.DataFrame:
    """Route points tagged by activity id, for drawing individual route lines."""
    frames = []
    sub = df[df["summary_polyline"].notna()]
    if max_routes is not None:
        sub = sub.head(max_routes)
    for _, r in sub.iterrows():
        pts = decode_polyline(r["summary_polyline"])[::every]
        if not pts:
            continue
        f = pd.DataFrame(pts, columns=["lat", "lon"])
        f["activity_id"] = r["id"]
        frames.append(f)
    if not frames:
        return pd.DataFrame(columns=["lat", "lon", "activity_id"])
    return pd.concat(frames, ignore_index=True)
