#!/bin/bash

start=$1
end=$2
job_index=$3

echo "Job index: $job_index, start: $start, end: $end"

## Inner loop from start to end
#for (( j=start; j<=end; j++ ))
#do
#   python 3D-tracking-scripts/track_objects_3d.py --data-shard $j --data-dir /ariesdv0/zhanling/oxe-data-converted
#   python 3D-tracking-scripts/save_3d_object_tracks.py --data-shard $j --data-dir /ariesdv0/zhanling/oxe-data-converted --tracking-3D-data-dir /ariesdv0/zhanling/oxe-data-converted/fractal20220817_depth_tracking_data/0.1.0
#done
python 3D-tracking-scripts/track_objects_3d.py --data-shard 508 --data-dir /ariesdv0/zhanling/oxe-data-converted
python 3D-tracking-scripts/save_3d_object_tracks.py --data-shard 508 --data-dir /ariesdv0/zhanling/oxe-data-converted --tracking-3D-data-dir /ariesdv0/zhanling/oxe-data-converted/fractal20220817_depth_tracking_data/0.1.0
python 3D-tracking-scripts/track_objects_3d.py --data-shard 634 --data-dir /ariesdv0/zhanling/oxe-data-converted
python 3D-tracking-scripts/save_3d_object_tracks.py --data-shard 634 --data-dir /ariesdv0/zhanling/oxe-data-converted --tracking-3D-data-dir /ariesdv0/zhanling/oxe-data-converted/fractal20220817_depth_tracking_data/0.1.0

