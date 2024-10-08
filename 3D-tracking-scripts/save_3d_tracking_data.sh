#!/bin/bash

start=$1
end=$2
job_index=$3

echo "Job index: $job_index, start: $start, end: $end"

# Inner loop from start to end
for (( j=start; j<=end; j++ ))
do
   python 3D-tracking-scripts/track_key_objects_pick.py --data-shard $j --data-dir /ariesdv0/zhanling/oxe-data-converted
   python 3D-tracking-scripts/save_tracking_data.py --data-shard $j --data-dir /ariesdv0/zhanling/oxe-data-converted --tracking-data-dir /ariesdv0/zhanling/oxe-data-converted/fractal20220817_3D_tracking_data/0.1.0
done