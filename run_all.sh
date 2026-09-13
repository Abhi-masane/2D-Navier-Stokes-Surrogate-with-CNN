#!/usr/bin/env bash
for caseDir in runs/case_Re_*; do
    echo "===== Running $caseDir ====="
    ( cd "$caseDir" && simpleFoam > log.simpleFoam 2>&1 \
        && foamToVTK -latestTime > log.foamToVTK 2>&1 ) \
        || echo "FAILED: $caseDir"
done
echo "All cases processed."
