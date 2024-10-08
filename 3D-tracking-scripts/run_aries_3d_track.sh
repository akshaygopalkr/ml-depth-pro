JOB="rt1-3D-track-key"
INSTALL="pip install -e . && source get_pretrained_models.sh"
RUN="bash 3D-tracking-scripts/save_3d_tracking_data.sh" 

for job_index in {0..0}
do  
    start=$(( (1024 / 4) * job_index ))
    end=$(( (1024 / 4) * (job_index + 1) -1))
    aries run ag-${JOB}-${job_index} -j 1 -g 1 lingzhan/openvla -- bash -c "nvidia-smi && git clone https://github.com/akshaygopalkr/ml-depth-pro.git && git pull && cd ml-depth-pro && ${INSTALL} && ${RUN} ${start} ${end} ${job_index}"
done