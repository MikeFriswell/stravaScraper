"""OAuth2 authentication for the Strava API.

One-time setup:
    1. Create an API application at https://www.strava.com/settings/api
       and set the "Authorization Callback Domain" to:  localhost
    2. Copy .env.example to .env and fill in your client id / secret.
    3. Run:  python strava_auth.py
       A browser opens; approve access. Tokens are saved to tokens.json.

After that the dashboard reads tokens.json and refreshes the access token
automatically (Strava access tokens expire every 6 hours).
"""
from __future__ import annotations

import http.server
import json
import os
import time
import urllib.parse
import webbrowser
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("STRAVA_CLIENT_ID")
CLIENT_SECRET = os.getenv("STRAVA_CLIENT_SECRET")

TOKENS_PATH = Path(__file__).with_name("tokens.json")

AUTH_URL = "https://www.strava.com/oauth/authorize"
TOKEN_URL = "https://www.strava.com/oauth/token"

# Must live under the "Authorization Callback Domain" registered on your app.
REDIRECT_HOST = "localhost"
REDIRECT_PORT = 8721
REDIRECT_URI = f"http://{REDIRECT_HOST}:{REDIRECT_PORT}/"

# activity:read_all also returns private activities; profile:read_all adds zones.
SCOPE = "read,activity:read_all,profile:read_all"


def _require_credentials() -> None:
    if not CLIENT_ID or not CLIENT_SECRET:
        raise SystemExit(
            "Missing STRAVA_CLIENT_ID / STRAVA_CLIENT_SECRET.\n"
            "Copy .env.example to .env and fill in the values from\n"
            "https://www.strava.com/settings/api"
        )


def _save_tokens(data: dict) -> None:
    TOKENS_PATH.write_text(json.dumps(data, indent=2))


def _load_tokens() -> dict | None:
    if TOKENS_PATH.exists():
        return json.loads(TOKENS_PATH.read_text())
    return None


def has_tokens() -> bool:
    return TOKENS_PATH.exists()


class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    """Captures the ?code=... that Strava sends to our redirect URI."""

    code: str | None = None
    error: str | None = None

    def do_GET(self):  # noqa: N802 (stdlib naming)
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        _CallbackHandler.code = qs.get("code", [None])[0]
        _CallbackHandler.error = qs.get("error", [None])[0]

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        msg = (
            "Strava authorised — you can close this tab and return to the terminal."
            if _CallbackHandler.code
            else f"Authorisation failed: {_CallbackHandler.error}"
        )
        self.wfile.write(f"<html><body><h2>{msg}</h2></body></html>".encode())

    def log_message(self, *args):  # silence the default request logging
        pass


def run_oauth_flow() -> dict:
    """Run the interactive browser authorization and persist the tokens."""
    _require_credentials()

    params = {
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "approval_prompt": "auto",
        "scope": SCOPE,
    }
    url = f"{AUTH_URL}?{urllib.parse.urlencode(params)}"

    print("Opening your browser to authorise access to Strava...")
    print("If it doesn't open, paste this URL manually:\n", url, "\n")

    server = http.server.HTTPServer((REDIRECT_HOST, REDIRECT_PORT), _CallbackHandler)
    webbrowser.open(url)
    while _CallbackHandler.code is None and _CallbackHandler.error is None:
        server.handle_request()
    server.server_close()

    if _CallbackHandler.error:
        raise SystemExit(f"Authorisation denied: {_CallbackHandler.error}")

    resp = requests.post(
        TOKEN_URL,
        data={
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "code": _CallbackHandler.code,
            "grant_type": "authorization_code",
        },
        timeout=30,
    )
    resp.raise_for_status()
    tokens = resp.json()
    _save_tokens(tokens)
    print(f"\nAuthorised! Tokens saved to {TOKENS_PATH.name}.")
    return tokens


def _refresh(refresh_token: str) -> dict:
    _require_credentials()
    resp = requests.post(
        TOKEN_URL,
        data={
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        },
        timeout=30,
    )
    resp.raise_for_status()
    tokens = resp.json()
    _save_tokens(tokens)
    return tokens


def get_valid_access_token() -> str:
    """Return a usable access token, refreshing it if it is about to expire."""
    tokens = _load_tokens()
    if not tokens:
        raise RuntimeError(
            "Not authorised yet. Run `python strava_auth.py` once to connect Strava."
        )
    # Refresh if the token has expired or expires within the next 5 minutes.
    if tokens.get("expires_at", 0) - time.time() < 300:
        tokens = _refresh(tokens["refresh_token"])
    return tokens["access_token"]


if __name__ == "__main__":
    run_oauth_flow()
