#!/usr/bin/env bash
# dl_setup.sh WORKSPACE BASE_ENV: isolated pipeline env on a shared GPU PC. Touches nothing outside WORKSPACE; no credentials.
# Their PyPI link is 30-70 KB/s (a torch install takes hours), so this copies a local torch env (read only) and adds the rest.
set -ex
W=$1; BASE=$2
mkdir -p "$W"/{logs,runs,data,models,hf,pipcache}
export PIP_CACHE_DIR=$W/pipcache
[ -x "$W/env/bin/python" ] || { rm -rf "$W/env.tmp"; cp -a "$BASE" "$W/env.tmp"; mv "$W/env.tmp" "$W/env"; }
cat > "$W/env/lib/python3.11/site-packages/sitecustomize.py" <<'PY'
# IPv6 to some CDNs hangs on these hosts; resolve IPv4 only.
import socket
_orig = socket.getaddrinfo
def _v4(host, port, family=0, *a, **k):
    return _orig(host, port, socket.AF_INET if family in (0, socket.AF_UNSPEC) else family, *a, **k)
socket.getaddrinfo = _v4
PY
"$W/env/bin/python" -m pip install -q --retries 10 --timeout 120 transformers==5.16.1 pycocotools kagglehub scipy
"$W/env/bin/python" -c "import torch,transformers,bitsandbytes as b;print(torch.__version__,torch.cuda.is_available(),torch.cuda.is_bf16_supported(),transformers.__version__,b.__version__)"
echo ENV_DONE
