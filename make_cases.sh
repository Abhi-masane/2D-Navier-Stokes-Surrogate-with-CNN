#!/usr/bin/env bash
set -e

BASE="$PWD/baseCase"
RUNS="$PWD/runs"
NU=0.01
DIAMETER=1.0
N_CASES=100
RE_START=5
RE_END=40

mkdir -p "$RUNS"

python3 - "$RE_START" "$RE_END" "$N_CASES" > Re_list.txt <<'PY'
import sys

start = float(sys.argv[1])
end = float(sys.argv[2])
count = int(sys.argv[3])

for i in range(count):
    value = start + i * (end - start) / (count - 1)
    print(f"{value:.6f}")
PY

while read -r reynolds; do
    case_name=$(printf "case_Re_%06.2f" "$reynolds")
    case_dir="$RUNS/$case_name"

    echo "creating $case_name"
    rm -rf "$case_dir"
    cp -r "$BASE" "$case_dir"

    inlet_velocity=$(python3 - "$reynolds" "$NU" "$DIAMETER" <<'PY'
import sys

reynolds = float(sys.argv[1])
nu = float(sys.argv[2])
diameter = float(sys.argv[3])
print(f"{reynolds * nu / diameter:.6f}")
PY
    )

    python3 scripts/set_inlet.py "$case_dir/0/U" "$inlet_velocity"
    echo "$reynolds" > "$case_dir/Re.txt"
    echo "$inlet_velocity" > "$case_dir/U.txt"
done < Re_list.txt

echo "created $N_CASES cases in $RUNS"
