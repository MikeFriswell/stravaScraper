"""Strava analysis dashboard.

Run with:  streamlit run app.py
(Authorise first with:  python strava_auth.py)
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

import analysis
import strava_auth
import strava_client

st.set_page_config(page_title="Strava Analysis", page_icon="🏃", layout="wide")

PLOT_TEMPLATE = "plotly_white"


def add_race_markers(fig, x_min=None):
    """Dashed vertical lines at each race date on a datetime-x chart.

    Plotly's add_vline annotation helper can't average string/Timestamp x's, so
    we pass the date as epoch-milliseconds (what date axes use internally).
    Pass ``x_min`` to extend the x-axis out to the races, so future race dates
    are visible (the empty space is your remaining training runway).
    """
    labels = {"10K": "🏰 10K", "Half-Marathon": "🌉 Half"}
    race_dates = [pd.Timestamp(r["date"]) for r in analysis.RACES]
    for r, d in zip(analysis.RACES, race_dates):
        fig.add_vline(
            x=d.value // 10**6, line_dash="dash", line_color="crimson", opacity=0.6,
            annotation_text=labels.get(r["predict_key"], r["name"]),
            annotation_position="top", annotation_font_color="crimson",
        )
    if x_min is not None and race_dates:
        fig.update_xaxes(range=[pd.Timestamp(x_min), max(race_dates) + pd.Timedelta(days=14)])
    return fig


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_enriched() -> pd.DataFrame:
    raw = strava_client.load_cached()
    if raw is None or raw.empty:
        return pd.DataFrame()
    return analysis.enrich(raw)


@st.cache_data(show_spinner=False)
def load_best_efforts() -> pd.DataFrame | None:
    return strava_client.load_best_efforts()


def sync_all(force_full: bool = False) -> None:
    """Pull new activities, then best efforts for any runs not yet scraped.

    Order matters: activities first, so the best-efforts step sees the new runs.
    The best-efforts step fetches a bounded chunk (no long UI-blocking sleeps);
    for a normal weekly top-up that's everything in one go, and after a big
    import you just click again to fetch the rest.
    """
    with st.spinner("Syncing activities and best efforts from Strava…"):
        try:
            strava_client.sync(force_full=force_full)
            strava_client.sync_best_efforts(
                max_requests=None, sleep_on_limit=False, log=lambda *a: None
            )
        except strava_client.RateLimitError as exc:
            st.error(str(exc))
            return
        except Exception as exc:  # noqa: BLE001 — surface any API/auth failure
            st.error(f"Sync failed: {exc}")
            return
    load_enriched.clear()
    load_best_efforts.clear()
    st.success("Synced activities + best efforts.")


# --------------------------------------------------------------------------- #
# Sidebar: connection, sync, units, filters
# --------------------------------------------------------------------------- #
st.sidebar.title("🏃 Strava Analysis")

if not strava_auth.has_tokens():
    st.sidebar.warning("Not connected to Strava yet.")
    st.title("Connect your Strava account")
    st.markdown(
        """
        This dashboard reads your activities from the **official Strava API**.

        **One-time setup**
        1. Create an API app at
           [strava.com/settings/api](https://www.strava.com/settings/api) and set
           the *Authorization Callback Domain* to `localhost`.
        2. Copy `.env.example` to `.env` and paste in your client id & secret.
        3. In a terminal, run:
           ```
           python strava_auth.py
           ```
           Approve access in the browser, then refresh this page.
        """
    )
    st.stop()

if st.sidebar.button("🔄 Sync everything", width="stretch"):
    sync_all()

_done, _total = strava_client.best_efforts_progress()
if _total:
    st.sidebar.caption(f"Best efforts scraped: {_done}/{_total} runs")
    if _done < _total:
        st.sidebar.caption("↳ click **Sync everything** again to fetch the rest.")

with st.sidebar.expander("Advanced"):
    if st.button("Full re-sync (rebuild activity cache)", width="stretch"):
        sync_all(force_full=True)

metric = st.sidebar.radio("Units", ["Metric (km)", "Imperial (mi)"], index=0) == "Metric (km)"
units = analysis.unit_columns(metric)
DIST, DIST_LABEL, DIST_UNIT = units["distance"]
ELEV, ELEV_LABEL, ELEV_UNIT = units["elevation"]
SPEED, SPEED_LABEL, _ = units["speed"]
PACE, PACE_LABEL, _ = units["pace"]

data = load_enriched()
if data.empty:
    st.info("No activities cached yet. Hit **Sync everything** in the sidebar to pull your data.")
    st.stop()

# Date-range filter.
min_d, max_d = data["date"].min(), data["date"].max()
date_range = st.sidebar.date_input(
    "Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_d, end_d = date_range
else:  # user mid-selection
    start_d, end_d = min_d, max_d

# Sport-type filter. Default to running activities so walks/rides don't skew
# the axes; other types can be re-added from the dropdown.
sports = sorted(data["sport_type"].dropna().unique().tolist())
run_default = [s for s in sports if s in analysis.RUN_SPORTS]
chosen_sports = st.sidebar.multiselect(
    "Activity types", sports, default=run_default or sports
)

mask = (
    (data["date"] >= start_d)
    & (data["date"] <= end_d)
    & (data["sport_type"].isin(chosen_sports))
)
df = data.loc[mask].copy()

st.sidebar.caption(f"{len(df):,} of {len(data):,} activities selected")

if df.empty:
    st.warning("No activities match the current filters.")
    st.stop()


# --------------------------------------------------------------------------- #
# Header KPIs
# --------------------------------------------------------------------------- #
st.title("Your Strava, analysed")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Activities", f"{len(df):,}")
k2.metric(f"Distance ({DIST_UNIT})", f"{df[DIST].sum():,.0f}")
k3.metric("Moving time", f"{df['moving_hr'].sum():,.0f} h")
k4.metric(f"Elevation ({ELEV_UNIT})", f"{df[ELEV].sum():,.0f}")
k5.metric("Relative effort", f"{df['suffer_score'].sum():,.0f}")

(tab_race, tab_vol, tab_pace, tab_pat, tab_fit, tab_con,
 tab_pb, tab_map, tab_hr) = st.tabs([
    "🎯 Races", "📈 Volume", "⚡ Pace & performance", "🗓️ Patterns",
    "📊 Fitness & form", "🔥 Consistency", "🏆 Records",
    "🗺️ Map", "❤️ HR & effort",
])


# --------------------------------------------------------------------------- #
# Races: countdown, predictions, goal pace/splits, endurance, goal-pace zones
# --------------------------------------------------------------------------- #
with tab_race:
    today = pd.Timestamp.today().normalize()
    unit_m = analysis.M_PER_KM if metric else analysis.M_PER_MILE
    pace_unit = "min/km" if metric else "min/mi"
    efforts = load_best_efforts()

    # Predictions from the last 12 months, using your fitted exponent if possible.
    fit = analysis.fit_fatigue_exponent(efforts, window_days=365)
    exp = fit[0] if fit else 1.06
    pred = analysis.predict_with_uncertainty(efforts, window_days=365, exponent=exp)
    pred_by_key = {row["race"]: row for _, row in pred.iterrows()} if not pred.empty else {}

    goal_paces: dict[str, float] = {}  # race name -> goal pace (sec per unit)

    for r in analysis.RACES:
        with st.container(border=True):
            d2g = analysis.days_until(r["date"], today)
            prow = pred_by_key.get(r["predict_key"])

            st.markdown(f"### {r['name']}")
            st.caption(f"{r['where']} · {pd.Timestamp(r['date']):%a %d %b %Y} · {r['notes']}")

            top = st.columns(3)
            top[0].metric("Days to go", f"{d2g}" if d2g >= 0 else "done")
            if prow is not None:
                top[1].metric(
                    "Predicted finish", analysis.format_duration(prow["predicted_s"]),
                    help=f"±{analysis.format_duration(prow['std_s'])} "
                         f"({analysis.format_duration(prow['low_s'])}–"
                         f"{analysis.format_duration(prow['high_s'])})",
                )
            else:
                top[1].metric("Predicted finish", "—")

            default_s = int(prow["predicted_s"]) if prow is not None else 3600
            g = st.columns([1, 1, 1, 3])
            gh = g[0].number_input("Goal h", 0, 9, default_s // 3600, key=f"{r['name']}_h")
            gm = g[1].number_input("min", 0, 59, (default_s % 3600) // 60, key=f"{r['name']}_m")
            gs = g[2].number_input("sec", 0, 59, default_s % 60, key=f"{r['name']}_s")
            goal_seconds = gh * 3600 + gm * 60 + gs
            pace_s = goal_seconds / (r["distance_m"] / unit_m) if goal_seconds else 0
            goal_paces[r["name"]] = pace_s
            top[2].metric(f"Goal pace ({pace_unit})", analysis.format_pace(pace_s / 60))

            if goal_seconds:
                with st.expander("Even-split plan"):
                    splits, _ = analysis.goal_splits(r["distance_m"], goal_seconds, unit_m)
                    tbl = pd.DataFrame({
                        DIST_UNIT: splits["mark"],
                        "Target clock": splits["cum_s"].apply(analysis.format_duration),
                    })
                    st.dataframe(tbl, hide_index=True, width="stretch")

    runs_only = df[df["sport_type"].isin(analysis.RUN_SPORTS)].copy()

    st.subheader("Long-run endurance build (for the half)")
    wl = analysis.weekly_longest_run(runs_only, DIST)
    half_units = 21097.5 / unit_m
    ten_units = 10000.0 / unit_m
    if wl.empty:
        st.info("No runs in the current filter.")
    else:
        recent = runs_only[runs_only["start_date_local"] >= today - pd.Timedelta(weeks=8)]
        recent_longest = recent[DIST].max()
        rl = float(recent_longest) if pd.notna(recent_longest) else 0.0
        e1, e2, e3 = st.columns(3)
        e1.metric(f"Longest run, last 8 wks ({DIST_UNIT})", f"{rl:.1f}")
        e2.metric(f"Gap to half ({DIST_UNIT})", f"{max(half_units - rl, 0):.1f}")
        e3.metric(f"Half distance ({DIST_UNIT})", f"{half_units:.1f}")

        fig = px.bar(
            wl, x="week", y="longest", template=PLOT_TEMPLATE,
            labels={"longest": f"Longest run ({DIST_UNIT})", "week": "Week"},
            title="Weekly longest run",
        )
        fig.add_hline(y=half_units, line_dash="dash", line_color="crimson",
                      annotation_text="Half", annotation_position="top left")
        fig.add_hline(y=ten_units, line_dash="dot", line_color="green",
                      annotation_text="10K", annotation_position="bottom left")
        add_race_markers(fig, x_min=wl["week"].min())
        st.plotly_chart(fig, width="stretch")
        st.caption("Aim to extend your long run toward ~18 km (≈90% of half distance), "
                   "with the longest 2–3 weeks out, then taper into race day.")

    st.subheader("Goal pace vs your training")
    recent_runs = runs_only[
        (runs_only["start_date_local"] >= today - pd.Timedelta(weeks=12))
        & runs_only[PACE].notna()
    ]
    if recent_runs.empty:
        st.info("No recent runs with pace data to compare.")
    else:
        fig = px.histogram(
            recent_runs, x=PACE, nbins=30, template=PLOT_TEMPLATE,
            labels={PACE: PACE_LABEL}, title="Your run paces, last 12 weeks",
        )
        for r in analysis.RACES:
            gp = goal_paces.get(r["name"])
            if gp:
                fig.add_vline(x=gp / 60, line_dash="dash", line_color="crimson",
                              annotation_text=f"{r['name']} goal",
                              annotation_position="top")
        st.plotly_chart(fig, width="stretch")
        tc = goal_paces.get("Two Castles 10K")
        if tc:
            easy_pct = float((recent_runs[PACE] > tc / 60).mean() * 100)
            st.caption(
                f"Relative to your 10K goal pace, **{easy_pct:.0f}%** of recent runs were "
                f"easier (slower) and **{100 - easy_pct:.0f}%** at/faster than goal. A "
                "polarised ~80/20 easy/hard split is the usual guideline."
            )


# --------------------------------------------------------------------------- #
# Volume trends
# --------------------------------------------------------------------------- #
with tab_vol:
    period = st.selectbox("Group by", ["week", "month", "year"], index=1)
    summary = analysis.period_summary(df, period, DIST)

    c1, c2 = st.columns(2)
    fig = px.bar(
        summary, x="period", y="distance", template=PLOT_TEMPLATE,
        labels={"distance": DIST_LABEL, "period": period.title()},
        title=f"Distance per {period}",
    )
    c1.plotly_chart(fig, width="stretch")

    summary = summary.sort_values("period")
    summary["cumulative"] = summary["distance"].cumsum()
    fig = px.area(
        summary, x="period", y="cumulative", template=PLOT_TEMPLATE,
        labels={"cumulative": f"Cumulative {DIST_UNIT}", "period": period.title()},
        title="Cumulative distance",
    )
    c2.plotly_chart(fig, width="stretch")

    c3, c4 = st.columns(2)
    fig = px.bar(
        summary, x="period", y="moving_hr", template=PLOT_TEMPLATE,
        labels={"moving_hr": "Moving time (h)", "period": period.title()},
        title=f"Moving time per {period}",
    )
    c3.plotly_chart(fig, width="stretch")

    fig = px.bar(
        summary, x="period", y="activities", template=PLOT_TEMPLATE,
        labels={"activities": "Activities", "period": period.title()},
        title=f"Activity count per {period}",
    )
    c4.plotly_chart(fig, width="stretch")

    st.subheader("Year-over-year")
    yoy = analysis.yearly_cumulative(df, DIST)
    if not yoy.empty:
        yoy["year"] = yoy["year"].astype(str)
        fig = px.line(
            yoy, x="doy", y="cumulative", color="year", template=PLOT_TEMPLATE,
            labels={"doy": "Day of year", "cumulative": f"Cumulative {DIST_UNIT}",
                    "year": "Year"},
            title="Cumulative distance by year (are you ahead of last year?)",
        )
        st.plotly_chart(fig, width="stretch")

        this_year = pd.Timestamp.today().year
        cur = yoy[yoy["year"] == str(this_year)]
        if not cur.empty:
            done = float(cur["cumulative"].max())
            doy = pd.Timestamp.today().dayofyear
            projected = done / doy * 365
            g1, g2, g3 = st.columns(3)
            g1.metric(f"{this_year} so far ({DIST_UNIT})", f"{done:,.0f}")
            g2.metric(f"Projected {this_year}", f"{projected:,.0f}")
            goal = g3.number_input(
                f"{this_year} goal ({DIST_UNIT})", min_value=0,
                value=int(round(projected / 100) * 100), step=50,
            )
            if goal > 0:
                st.progress(
                    min(done / goal, 1.0),
                    text=f"{done:,.0f} / {goal:,.0f} {DIST_UNIT} ({done / goal * 100:.0f}%)",
                )


# --------------------------------------------------------------------------- #
# Pace & performance
# --------------------------------------------------------------------------- #
with tab_pace:
    foot = df[df["sport_type"].isin(analysis.FOOT_SPORTS) & df[PACE].notna()].copy()
    wheels = df[~df["sport_type"].isin(analysis.FOOT_SPORTS) & df[SPEED].notna()].copy()

    if not foot.empty:
        st.subheader("Running / walking pace")
        c1, c2 = st.columns(2)

        foot = foot.sort_values("start_date_local")
        foot["pace_roll"] = foot[PACE].rolling(10, min_periods=3).mean()
        fig = px.scatter(
            foot, x="start_date_local", y=PACE, color="sport_type",
            template=PLOT_TEMPLATE, opacity=0.6,
            labels={PACE: PACE_LABEL, "start_date_local": "Date"},
            title="Pace over time (lower is faster)",
        )
        fig.add_scatter(
            x=foot["start_date_local"], y=foot["pace_roll"],
            mode="lines", name="10-activity avg", line=dict(width=3),
        )
        fig.update_yaxes(autorange="reversed")
        c1.plotly_chart(fig, width="stretch")

        fig = px.scatter(
            foot, x=DIST, y=PACE, color="year", template=PLOT_TEMPLATE,
            labels={DIST: DIST_LABEL, PACE: PACE_LABEL},
            title="Pace vs distance", color_continuous_scale="Viridis",
        )
        fig.update_yaxes(autorange="reversed")
        c2.plotly_chart(fig, width="stretch")

    if not wheels.empty:
        st.subheader("Cycling / other speed")
        wheels = wheels.sort_values("start_date_local")
        wheels["speed_roll"] = wheels[SPEED].rolling(10, min_periods=3).mean()
        fig = px.scatter(
            wheels, x="start_date_local", y=SPEED, color="sport_type",
            template=PLOT_TEMPLATE, opacity=0.6,
            labels={SPEED: SPEED_LABEL, "start_date_local": "Date"},
            title="Average speed over time",
        )
        fig.add_scatter(
            x=wheels["start_date_local"], y=wheels["speed_roll"],
            mode="lines", name="10-activity avg", line=dict(width=3),
        )
        st.plotly_chart(fig, width="stretch")

    if foot.empty and wheels.empty:
        st.info("No pace/speed data available for the selected activities.")


# --------------------------------------------------------------------------- #
# Training patterns
# --------------------------------------------------------------------------- #
with tab_pat:
    c1, c2 = st.columns(2)

    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    by_dow = df.groupby("day_of_week").size().reindex(dow_order, fill_value=0).reset_index()
    by_dow.columns = ["day_of_week", "activities"]
    fig = px.bar(
        by_dow, x="day_of_week", y="activities", template=PLOT_TEMPLATE,
        labels={"day_of_week": "", "activities": "Activities"},
        title="When do you train? (day of week)",
    )
    c1.plotly_chart(fig, width="stretch")

    by_hour = df.groupby("hour").size().reindex(range(24), fill_value=0).reset_index()
    by_hour.columns = ["hour", "activities"]
    fig = px.bar(
        by_hour, x="hour", y="activities", template=PLOT_TEMPLATE,
        labels={"hour": "Hour of day", "activities": "Activities"},
        title="Time of day",
    )
    c2.plotly_chart(fig, width="stretch")

    c3, c4 = st.columns(2)
    by_type = (
        df.groupby("sport_type")
        .agg(activities=("id", "count"), distance=(DIST, "sum"))
        .reset_index()
        .sort_values("activities", ascending=False)
    )
    fig = px.pie(
        by_type, names="sport_type", values="activities",
        template=PLOT_TEMPLATE, title="Activity mix (by count)", hole=0.4,
    )
    c3.plotly_chart(fig, width="stretch")

    # Day x hour heatmap of activity counts.
    heat = (
        df.assign(dow=pd.Categorical(df["day_of_week"], categories=dow_order, ordered=True))
        .pivot_table(index="dow", columns="hour", values="id", aggfunc="count", fill_value=0,
                     observed=False)
    )
    fig = px.imshow(
        heat, template=PLOT_TEMPLATE, aspect="auto", color_continuous_scale="Blues",
        labels={"x": "Hour", "y": "", "color": "Activities"},
        title="Heatmap: day × hour",
    )
    c4.plotly_chart(fig, width="stretch")


# --------------------------------------------------------------------------- #
# Fitness & Form (training load)
# --------------------------------------------------------------------------- #
with tab_fit:
    st.caption(
        "Training load is modelled from Strava's relative effort, with a "
        "duration-based estimate where effort wasn't recorded. **Fitness** is a "
        "~42-day average of load, **Fatigue** a ~7-day average, and **Form** = "
        "Fitness − Fatigue."
    )
    tl = analysis.training_load(df)
    if tl.empty:
        st.info("Not enough data for the selected activities.")
    else:
        latest = tl.dropna(subset=["form"]).iloc[-1]
        m1, m2, m3 = st.columns(3)
        m1.metric("Fitness (CTL)", f"{latest['fitness']:.0f}")
        m2.metric("Fatigue (ATL)", f"{latest['fatigue']:.0f}")
        m3.metric("Form (TSB)", f"{latest['form']:+.0f}")

        melt = tl.melt(
            id_vars="date", value_vars=["fitness", "fatigue", "form"],
            var_name="metric", value_name="value",
        )
        metric_labels = {"fitness": "Fitness", "fatigue": "Fatigue", "form": "Form"}
        melt["metric"] = melt["metric"].map(metric_labels)
        fig = px.line(
            melt, x="date", y="value", color="metric", template=PLOT_TEMPLATE,
            labels={"value": "", "date": "Date", "metric": ""},
            title="Fitness, Fatigue & Form",
        )
        add_race_markers(fig, x_min=tl["date"].min())
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Positive form = fresh/tapered (good for racing); strongly negative form "
            "means you're carrying fatigue — productive in a training block, risky if "
            "sustained. Rising fitness over weeks is the goal."
        )

        fig = px.bar(
            tl, x="date", y="load", template=PLOT_TEMPLATE,
            labels={"load": "Daily load", "date": "Date"}, title="Daily training load",
        )
        st.plotly_chart(fig, width="stretch")


# --------------------------------------------------------------------------- #
# Consistency: streaks, calendar heatmap, workload ratio
# --------------------------------------------------------------------------- #
with tab_con:
    target = int(st.number_input("Runs per week target", min_value=1, max_value=14,
                                 value=3, step=1))
    runs_df = df[df["sport_type"].isin(analysis.RUN_SPORTS)]
    wc, cstats = analysis.weekly_consistency(runs_df, target=target)

    if not cstats:
        st.info("No runs in the current filter to assess weekly consistency.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Avg runs / week", f"{cstats['avg_per_week']:.1f}")
        m2.metric("Weeks on target", f"{cstats['weeks_met']}/{cstats['weeks_total']}",
                  f"{cstats['pct_met']:.0f}%")
        m3.metric("Current streak", f"{cstats['current_streak']} wks")
        m4.metric("Longest streak", f"{cstats['longest_streak']} wks")
        st.caption(
            f"A week is **on target** at {target}+ runs. Weeks below target still "
            "count toward your average — they just don't extend the streak. The "
            "in-progress week is shown but excluded from streak/percentage stats."
        )

        fig = px.bar(
            wc, x="week", y="runs", color="status",
            category_orders={"status": ["Target met", "Partial", "Rest week", "This week"]},
            color_discrete_map={"Target met": "#2e7d32", "Partial": "#f9a825",
                                "Rest week": "#e0e0e0", "This week": "#1e88e5"},
            template=PLOT_TEMPLATE,
            labels={"runs": "Runs", "week": "Week", "status": ""},
            title=f"Runs per week (target: {target})",
        )
        fig.add_hline(y=target, line_dash="dash", line_color="crimson",
                      annotation_text="Target", annotation_position="top left")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Calendar heatmap")
    years = sorted(df["year"].unique())
    yr = int(st.selectbox("Year", years, index=len(years) - 1))
    matrix, ticks = analysis.calendar_matrix(df, yr, DIST)
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig = px.imshow(
        matrix.values, template=PLOT_TEMPLATE, aspect="auto",
        color_continuous_scale="Greens",
        labels={"x": "", "y": "", "color": DIST_UNIT},
        title=f"{yr}: daily distance",
    )
    fig.update_yaxes(tickvals=list(range(7)), ticktext=dow_labels)
    fig.update_xaxes(tickvals=[t[0] for t in ticks], ticktext=[t[1] for t in ticks])
    fig.update_layout(coloraxis_showscale=True)
    st.plotly_chart(fig, width="stretch")

    st.subheader("Weekly load & injury-risk ratio")
    c1, c2 = st.columns(2)
    wk = df.groupby("week")[DIST].sum().reset_index()
    fig = px.bar(
        wk, x="week", y=DIST, template=PLOT_TEMPLATE,
        labels={DIST: DIST_LABEL, "week": "Week"}, title="Weekly distance",
    )
    c1.plotly_chart(fig, width="stretch")

    acwr = analysis.workload_ratio(df, DIST)
    fig = px.line(
        acwr, x="date", y="acwr", template=PLOT_TEMPLATE,
        labels={"acwr": "Acute : chronic ratio", "date": "Date"},
        title="Workload ratio (ACWR)",
    )
    fig.add_hrect(y0=0.8, y1=1.3, fillcolor="green", opacity=0.12, line_width=0)
    fig.add_hline(y=1.5, line_dash="dot", line_color="red")
    add_race_markers(fig, x_min=acwr["date"].min())
    c2.plotly_chart(fig, width="stretch")
    st.caption(
        "Shaded band (0.8–1.3) is the commonly cited ACWR 'sweet spot'; spikes above "
        "~1.5 reflect rapid load increases associated with higher injury risk."
    )


# --------------------------------------------------------------------------- #
# Records & personal bests
# --------------------------------------------------------------------------- #
with tab_pb:
    def _record(frame: pd.DataFrame, col: str, largest: bool = True):
        valid = frame[frame[col].notna()]
        if valid.empty:
            return None
        row = valid.loc[valid[col].idxmax() if largest else valid[col].idxmin()]
        return row

    st.subheader("All-time bests (within current filters)")

    longest = _record(df, DIST)
    longest_t = _record(df, "moving_time")
    climb = _record(df, ELEV)
    effort = _record(df, "suffer_score")

    # Fastest pace, only counting runs over 1 km / 0.6 mi to avoid sprints skewing.
    runs = df[df["sport_type"].isin(analysis.FOOT_SPORTS) & (df["distance"] > 1000)]
    fastest = _record(runs, PACE, largest=False) if not runs.empty else None

    fast_speed = _record(df, SPEED)

    def show(colobj, label, row, value_fmt):
        if row is None:
            colobj.metric(label, "—")
            return
        colobj.metric(label, value_fmt(row))
        colobj.caption(f"{row['name']} · {pd.to_datetime(row['start_date_local']):%d %b %Y}")

    r1c1, r1c2, r1c3 = st.columns(3)
    show(r1c1, f"Longest distance ({DIST_UNIT})", longest, lambda r: f"{r[DIST]:.1f}")
    show(r1c2, "Longest duration", longest_t, lambda r: analysis.format_duration(r["moving_time"]))
    show(r1c3, f"Biggest climb ({ELEV_UNIT})", climb, lambda r: f"{r[ELEV]:,.0f}")

    r2c1, r2c2, r2c3 = st.columns(3)
    show(r2c1, f"Fastest pace ({PACE_LABEL.split('(')[-1].rstrip(')')})", fastest,
         lambda r: analysis.format_pace(r[PACE]))
    show(r2c2, SPEED_LABEL.replace("Speed", "Top avg speed"), fast_speed, lambda r: f"{r[SPEED]:.1f}")
    show(r2c3, "Highest relative effort", effort, lambda r: f"{r['suffer_score']:.0f}")

    st.subheader("All-time totals")
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Activities", f"{len(df):,}")
    t2.metric(f"Distance ({DIST_UNIT})", f"{df[DIST].sum():,.0f}")
    t3.metric("Moving time", f"{df['moving_hr'].sum():,.0f} h")
    t4.metric(f"Elevation ({ELEV_UNIT})", f"{df[ELEV].sum():,.0f}")

    st.subheader("Top 10 longest activities")
    top = df.nlargest(10, DIST)[
        ["start_date_local", "name", "sport_type", DIST, "moving_time", ELEV, PACE]
    ].copy()
    top["start_date_local"] = pd.to_datetime(top["start_date_local"]).dt.strftime("%d %b %Y")
    top["moving_time"] = top["moving_time"].apply(analysis.format_duration)
    top[PACE] = top[PACE].apply(analysis.format_pace)
    top = top.rename(columns={
        "start_date_local": "Date", "name": "Activity", "sport_type": "Type",
        DIST: DIST_LABEL, "moving_time": "Time", ELEV: ELEV_LABEL, PACE: PACE_LABEL,
    })
    st.dataframe(top, width="stretch", hide_index=True)

    st.divider()
    st.subheader("Personal bests (fastest efforts within your runs)")
    done, total = strava_client.best_efforts_progress()
    if total and done < total:
        st.caption(f"{done} of {total} runs scraped — click **Sync everything** in the "
                   "sidebar to fetch the rest.")

    prs = analysis.best_effort_prs(load_best_efforts())
    if prs.empty:
        st.info("No best-effort data yet — click **Sync everything** in the sidebar.")
    else:
        unit_m = analysis.M_PER_KM if metric else analysis.M_PER_MILE
        pr_table = prs.copy()
        pr_table["Best time"] = pr_table["elapsed_time"].apply(analysis.format_duration)
        pr_table["Pace"] = (
            (pr_table["elapsed_time"] / 60) / (pr_table["distance"] / unit_m)
        ).apply(analysis.format_pace)
        pr_table["Date"] = pr_table["start_date_local"].dt.strftime("%d %b %Y")
        pr_table = pr_table.rename(
            columns={"name": "Distance", "activity_name": "Activity",
                     "Pace": PACE_LABEL}
        )
        st.dataframe(
            pr_table[["Distance", "Best time", PACE_LABEL, "Date", "Activity"]],
            width="stretch", hide_index=True,
        )

        st.subheader("Race-time predictor")
        st.caption(
            "Each prediction blends Riegel extrapolations from **all** your best "
            "efforts, weighted toward distances near the target. The **± range** is "
            "the weighted standard deviation — a wider band means your efforts "
            "disagree more, so treat the estimate with more caution."
        )
        windows = {"Last 3 months": 90, "Last 6 months": 182, "Last 12 months": 365,
                   "Last 2 years": 730, "All time": None}
        ecol, xcol = st.columns(2)
        wlabel = ecol.selectbox("Base on efforts from", list(windows), index=2)
        exp_mode = xcol.radio(
            "Fatigue exponent", ["Riegel (1.06)", "Fit to my data"], horizontal=True,
        )
        if exp_mode.startswith("Fit"):
            fit = analysis.fit_fatigue_exponent(load_best_efforts(), windows[wlabel])
            if fit is None:
                st.warning("Not enough distinct distances (≥1 km) in this window to "
                           "fit an exponent — using Riegel's 1.06.")
                exponent = 1.06
            else:
                exponent, r2, npts = fit
                st.caption(
                    f"Your fitted fatigue exponent is **{exponent:.3f}** "
                    f"(R²={r2:.2f} over {npts} distances ≥1 km). Riegel's default is "
                    "1.06 — higher means you fade more with distance, lower means you "
                    "hold pace better as runs get longer."
                )
        else:
            exponent = 1.06
        pred = analysis.predict_with_uncertainty(
            load_best_efforts(), window_days=windows[wlabel], exponent=exponent
        )
        if pred.empty:
            st.info(f"No best efforts within '{wlabel}'. Try a wider window.")
        else:
            n_basis = int(pred["n_efforts"].max())
            show = pd.DataFrame({
                "Distance": pred["race"],
                "Predicted": pred["predicted_s"].apply(analysis.format_duration),
                "Range (± spread)": [
                    f"{analysis.format_duration(lo)} – {analysis.format_duration(hi)}"
                    for lo, hi in zip(pred["low_s"], pred["high_s"])
                ],
            })
            st.dataframe(show, width="stretch", hide_index=True)

            fig = px.bar(
                pred.assign(pred_min=pred["predicted_s"] / 60,
                            err_min=pred["std_s"] / 60),
                x="race", y="pred_min", error_y="err_min", template=PLOT_TEMPLATE,
                labels={"race": "", "pred_min": "Predicted time (min)"},
                title="Predicted times with uncertainty (± weighted std)",
            )
            st.plotly_chart(fig, width="stretch")
            st.caption(f"Based on your fastest effort at each of {n_basis} distance(s) "
                       f"within {wlabel.lower()}.")


# --------------------------------------------------------------------------- #
# Map / route heatmap
# --------------------------------------------------------------------------- #
with tab_map:
    n_routes = int(df["summary_polyline"].notna().sum())
    if n_routes == 0:
        st.info("No route data for these activities. Use **Full re-sync** in the "
                "sidebar to capture route shapes.")
    else:
        view = st.radio("View", ["Heatmap", "Route lines"], horizontal=True)
        if view == "Heatmap":
            pts = analysis.route_points(df)
            if pts.empty:
                st.info("No route points to plot.")
            else:
                fig = px.density_map(
                    pts, lat="lat", lon="lon", radius=4,
                    center={"lat": float(pts["lat"].median()),
                            "lon": float(pts["lon"].median())},
                    zoom=7, map_style="open-street-map",
                    title=f"Route heatmap · {n_routes} activities",
                )
                fig.update_layout(height=620, margin={"l": 0, "r": 0, "t": 40, "b": 0})
                st.plotly_chart(fig, width="stretch")
        else:
            max_routes = st.slider(
                "Max routes to draw", 10, min(300, n_routes), min(100, n_routes)
            )
            lines = analysis.route_lines(df, max_routes=max_routes)
            if lines.empty:
                st.info("No route points to plot.")
            else:
                lines["activity_id"] = lines["activity_id"].astype(str)
                fig = px.line_map(
                    lines, lat="lat", lon="lon", line_group="activity_id",
                    center={"lat": float(lines["lat"].median()),
                            "lon": float(lines["lon"].median())},
                    zoom=7, map_style="open-street-map",
                    title=f"{lines['activity_id'].nunique()} routes",
                )
                fig.update_traces(line={"width": 1}, opacity=0.5)
                fig.update_layout(height=620, margin={"l": 0, "r": 0, "t": 40, "b": 0})
                st.plotly_chart(fig, width="stretch")


# --------------------------------------------------------------------------- #
# Heart rate & effort
# --------------------------------------------------------------------------- #
with tab_hr:
    hr = df[df["average_heartrate"].notna()].copy()
    if hr.empty:
        st.info("No heart-rate data in the selected activities.")
    else:
        # Aerobic efficiency trend: metres/min per heartbeat (higher = fitter).
        ef = hr[hr["sport_type"].isin(analysis.RUN_SPORTS) & hr["aerobic_ef"].notna()]
        ef = ef.sort_values("start_date_local")
        if not ef.empty:
            ef = ef.assign(ef_roll=ef["aerobic_ef"].rolling(10, min_periods=3).mean())
            fig = px.scatter(
                ef, x="start_date_local", y="aerobic_ef", template=PLOT_TEMPLATE,
                opacity=0.5,
                labels={"aerobic_ef": "Efficiency (m/min per bpm)",
                        "start_date_local": "Date"},
                title="Aerobic efficiency trend (higher = more speed per heartbeat)",
            )
            fig.add_scatter(x=ef["start_date_local"], y=ef["ef_roll"], mode="lines",
                            name="10-run avg", line={"width": 3})
            st.plotly_chart(fig, width="stretch")

        c1, c2 = st.columns(2)

        # Pace vs HR scatter for foot sports.
        foot_hr = hr[hr["sport_type"].isin(analysis.FOOT_SPORTS) & hr[PACE].notna()]
        if not foot_hr.empty:
            fig = px.scatter(
                foot_hr, x="average_heartrate", y=PACE, color="year",
                template=PLOT_TEMPLATE, color_continuous_scale="Plasma",
                labels={"average_heartrate": "Avg HR (bpm)", PACE: PACE_LABEL},
                title="Pace vs heart rate (lower-left = more efficient)",
            )
            fig.update_yaxes(autorange="reversed")
            c1.plotly_chart(fig, width="stretch")
        else:
            c1.info("No pace+HR foot activities to compare.")

        # HR trend over time.
        hr_sorted = hr.sort_values("start_date_local")
        fig = px.scatter(
            hr_sorted, x="start_date_local", y="average_heartrate", color="sport_type",
            template=PLOT_TEMPLATE, opacity=0.6,
            labels={"average_heartrate": "Avg HR (bpm)", "start_date_local": "Date"},
            title="Average heart rate over time",
        )
        c2.plotly_chart(fig, width="stretch")

        c3, c4 = st.columns(2)
        fig = px.histogram(
            hr, x="average_heartrate", nbins=30, template=PLOT_TEMPLATE,
            labels={"average_heartrate": "Avg HR (bpm)"},
            title="Distribution of average heart rate",
        )
        c3.plotly_chart(fig, width="stretch")

        effort = df[df["suffer_score"].notna()].sort_values("start_date_local")
        if not effort.empty:
            fig = px.scatter(
                effort, x="start_date_local", y="suffer_score", color=DIST,
                template=PLOT_TEMPLATE, color_continuous_scale="Inferno",
                labels={"suffer_score": "Relative effort", "start_date_local": "Date",
                        DIST: DIST_LABEL},
                title="Relative effort (suffer score) over time",
            )
            c4.plotly_chart(fig, width="stretch")
        else:
            c4.info("No relative-effort data available.")

        # Power, when present (cycling).
        pwr = df[df["average_watts"].notna()]
        if not pwr.empty:
            st.subheader("Power (cycling)")
            fig = px.scatter(
                pwr.sort_values("start_date_local"), x="start_date_local",
                y="average_watts", color=SPEED, template=PLOT_TEMPLATE,
                labels={"average_watts": "Avg power (W)", "start_date_local": "Date",
                        SPEED: SPEED_LABEL},
                title="Average power over time",
            )
            st.plotly_chart(fig, width="stretch")
