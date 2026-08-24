#!/bin/bash
# Extract the USER_DATA heredoc from a launch script and syntax-check it as the
# shell script it will actually become on the instance. bash -n on the launch
# script itself does NOT do this: user-data is quoted text and is never parsed.
set -euo pipefail
SCRIPT="$1"
python3 - "$SCRIPT" > /tmp/userdata_extracted.sh <<'PY'
import re, sys
s = open(sys.argv[1]).read()
m = re.search(r"USER_DATA=\$\(cat <<'USERDATA'\n(.*?)\nUSERDATA\n\)", s, re.S)
if not m:
    print("could not extract USER_DATA", file=sys.stderr); sys.exit(2)
body = m.group(1)
# Substitute placeholders with harmless literals so it parses.
for ph in set(re.findall(r"__[A-Z0-9_]+__", body)):
    body = body.replace(ph, "PLACEHOLDER")
print(body)
PY
bash -n /tmp/userdata_extracted.sh && echo "USER-DATA SYNTAX OK ($(wc -l < /tmp/userdata_extracted.sh) lines)"

# `aws ec2 run-instances --user-data` accepts raw text and performs the API's
# required base64 encoding itself. Pre-encoding here produces a syntactically
# valid launch script but cloud-init receives inert base64 text.
if grep -q 'USER_DATA_B64' "$SCRIPT"; then
    echo "user-data must not be pre-encoded before aws ec2 run-instances" >&2
    exit 3
fi
grep -Fq -- '--user-data "$USER_DATA"' "$SCRIPT" || {
    echo 'launch script must pass raw $USER_DATA to aws ec2 run-instances' >&2
    exit 4
}
echo "USER-DATA AWS ENCODING OK"
