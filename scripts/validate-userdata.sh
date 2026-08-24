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

# Inspect the argument on the real RunInstances command, excluding comments.
# Requiring the direct raw variable rejects pre-encoding regardless of the
# intermediary variable's name.
commands = "\n".join(line for line in s.splitlines() if not line.lstrip().startswith("#"))
args = re.findall(r'--user-data\s+("[^"]*"|\'[^\']*\'|\S+)', commands)
if args != ['"$USER_DATA"']:
    print(
        f'aws ec2 run-instances must pass raw "$USER_DATA" exactly once; found {args}',
        file=sys.stderr,
    )
    sys.exit(3)
print(body)
PY
bash -n /tmp/userdata_extracted.sh && echo "USER-DATA SYNTAX OK ($(wc -l < /tmp/userdata_extracted.sh) lines)"
echo "USER-DATA AWS ENCODING OK"
