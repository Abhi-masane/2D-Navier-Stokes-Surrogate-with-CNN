#!/usr/bin/env bash
set -e
BASE=$PWD/baseCase
RUNS=$PWD/runs
NU=0.01
D=1.0
N=100
RE_START=5
RE_END=40
mkdir -p "$RUNS"
python3 -c "
import sys
start = float(sys.argv[1]); end = float(sys.argv[2]); n = int(sys.argv[3])
for i in range(n):
    print(f'{start + i*(end-start)/(n-1):.6f}')
" "$RE_START" "$RE_END" "$N" > Re_list.txt
while read Re; do
    caseName=$(printf "case_Re_%06.2f" "$Re")
    dst="$RUNS/$caseName"
    echo "Creating $caseName"
    rm -rf "$dst"
    cp -r "$BASE" "$dst"
    U=$(python3 -c "
import sys
Re = float(sys.argv[1]); nu = float(sys.argv[2]); D = float(sys.argv[3])
print(f'{Re*nu/D:.6f}')
" "$Re" "$NU" "$D")
    python3 scripts/set_inlet.py "$dst/0/U" "$U"
    echo "$Re" > "$dst/Re.txt"
    echo "$U" > "$dst/U.txt"
done < Re_list.txt
echo "Created $N cases in $RUNS"
