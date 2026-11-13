"""Run the Kaggle CLI as any account listed in Secrets.env.

    python Scripts/kgl.py list                      # aliases + usernames
    python Scripts/kgl.py check                     # verify every token works
    python Scripts/kgl.py K01 kernels status user13/vistaqa-shard0
    python Scripts/kgl.py K01 kernels push -p kernels/shard0
    python Scripts/kgl.py upload DIR SLUG K01 K02 ...   # private dataset on each account

Credentials go into the child process env only; ~/.kaggle is never touched.
"""
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SECRETS = Path(__file__).resolve().parent.parent / "Secrets.env"


def load_accounts():
    kv = {}
    for line in SECRETS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip()
    accts = {}
    for k, user in kv.items():
        if k.startswith("KAGGLE_USERNAME_"):
            alias = k[len("KAGGLE_USERNAME_"):]
            token = kv.get("KAGGLE_API_TOKEN_" + alias, "")
            if user and token:
                accts[alias.upper()] = (user, token)
    return accts


def env_for(user, token):
    env = {k: v for k, v in os.environ.items() if not k.startswith("KAGGLE_")}
    # kagglesdk falls back to ~/.kaggle/access_token (someone else's account); point ~ at an empty dir.
    home = Path(__file__).resolve().parent / ".kaggle_home"
    (home / ".kaggle").mkdir(parents=True, exist_ok=True)
    env["HOME"] = env["USERPROFILE"] = str(home)
    env["KAGGLE_CONFIG_DIR"] = str(home / ".kaggle")
    env["KAGGLE_USERNAME"] = user
    if token.startswith("KGAT_"):
        env["KAGGLE_API_TOKEN"] = token
    else:  # legacy kaggle.json-style key
        env["KAGGLE_KEY"] = token
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"  # CLI writes kernel logs with the locale codec otherwise (cp1252 -> empty log)
    return env


def run(alias, args, capture=False, tries=4):
    user, token = load_accounts()[alias.upper()]
    for i in range(tries):  # Kaggle API drops connections now and then (SSL EOF / max retries)
        r = subprocess.run(["kaggle", *args], env=env_for(user, token),
                           capture_output=capture, text=True, encoding="utf-8")
        flaky = capture and any(s in (r.stdout or "") + (r.stderr or "") for s in ("SSLError", "Max retries", "ConnectionError", "503", "502"))
        if not flaky:
            return r
        time.sleep(5 * (i + 1))
    return r


def upload(alias, folder, slug):
    """Private dataset on this account through kagglehub (`kaggle datasets create` fails on Windows); new version if it exists."""
    user, token = load_accounts()[alias.upper()]
    who = next(l.split(":", 1)[1].strip() for l in run(alias, ["config", "view"], capture=True).stdout.splitlines() if "username:" in l)
    code = f"import kagglehub; kagglehub.dataset_upload({who + '/' + slug!r}, {folder!r}, version_notes='vlm4rwd')"
    r = subprocess.run([sys.executable, "-c", code], env=env_for(user, token), capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(alias, who, "ok" if r.returncode == 0 else (r.stdout + r.stderr).strip()[-300:], flush=True)


def check(alias):
    user, _ = load_accounts()[alias]
    r = run(alias, ["config", "view"], capture=True)
    who = [l.split(":", 1)[1].strip() for l in r.stdout.splitlines() if "username:" in l]
    r2 = run(alias, ["kernels", "list", "--mine", "--page-size", "1"], capture=True)
    ok = r2.returncode == 0 and bool(who)
    note = "" if who == [user] else f"token belongs to {who} (use that in kernel ids)" if ok else (r2.stderr or r2.stdout).strip()[:80]
    return alias, user, ok, note


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cmd = sys.argv[1]
    accts = load_accounts()
    if cmd == "list":
        for a, (u, _) in accts.items():
            print(f"{a:14s} {u}")
    elif cmd == "check":
        with ThreadPoolExecutor(8) as ex:
            for a, u, ok, err in ex.map(check, accts):
                print(f"{'OK ' if ok else 'BAD'} {a:14s} {u} {err or ''}")
    elif cmd == "upload":
        for a in sys.argv[4:]:
            upload(a, sys.argv[2], sys.argv[3])
    else:
        sys.exit(run(cmd, sys.argv[2:]).returncode)
