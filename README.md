# Strava Analysis Dashboard

Pull your own activities from the **official Strava API** and explore them in an
interactive [Streamlit](https://streamlit.io) dashboard — volume trends, pace &
performance, training patterns, personal records, and heart-rate / effort
comparisons.

> Replaces the original Selenium web-scraper (`scraper.py`), which broke whenever
> Strava changed its site. The API approach is robust, returns clean JSON, and is
> within Strava's terms of service.

## What you get

- **Volume** – distance, time, elevation and activity count by week / month / year, plus cumulative totals.
- **Pace & performance** – pace-over-time with a rolling average, pace vs distance, cycling speed trends.
- **Patterns** – day-of-week and time-of-day habits, activity-type mix, and a day×hour heatmap.
- **Records** – longest/biggest efforts, fastest pace, top relative-effort, all-time totals, top-10 table.
- **HR & effort** – aerobic efficiency (pace vs HR), HR trends and distribution, relative effort, and power for rides.

Switch between metric and imperial units, filter by date range and activity type.

## Setup

### 1. Create a Strava API application
Go to <https://www.strava.com/settings/api> and create an app. Set:

- **Authorization Callback Domain:** `localhost`

Note the **Client ID** and **Client Secret**.

### 2. Install
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add your credentials
```powershell
copy .env.example .env
```
Edit `.env` and paste in your client id and secret.

### 4. Authorise (one time)
```powershell
python strava_auth.py
```
A browser opens — approve access. Tokens are saved to `tokens.json` and refreshed
automatically afterwards.

### 5. Run the dashboard
```powershell
streamlit run app.py
```
Click **Sync new** in the sidebar to download your activities (cached to
`data/activities.json`). The first sync of a long history may take a minute.

## How it fits together

| File | Role |
|------|------|
| `strava_auth.py` | OAuth2: one-time browser login + automatic token refresh |
| `strava_client.py` | Calls the Strava API, caches activities, incremental sync |
| `analysis.py` | Pandas transforms + unit handling + metrics |
| `app.py` | The Streamlit dashboard |

## Notes

- **Privacy:** `.env`, `tokens.json`, and `data/` hold your credentials and personal
  data and are git-ignored. Don't commit them.
- **Rate limits:** Strava allows ~100 requests / 15 min by default. Syncs are
  incremental (only new activities), so you'll rarely get close.
- **Scope of data:** the dashboard uses Strava's *summary* activity fields, which
  cover everything above. Segment-level PRs and per-second streams would need
  extra per-activity API calls and could be added later.
