#!/usr/bin/env bash

for case_dir in runs/case_Re_*; do
    [ -d "$case_dir" ] || continue

    echo "===== running $case_dir ====="

    if (
        cd "$case_dir" &&
        simpleFoam > log.simpleFoam 2>&1 &&
        foamToVTK -latestTime > log.foamToVTK 2>&1
    ); then
        echo "finished $case_dir"
    else
        echo "failed $case_dir"
    fi
done

echo "all cases processed"
