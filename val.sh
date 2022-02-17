EXPERIMENT="Garbage_yolov5s_attn_lite_aug_ar0005_16-02-2022"
#runs/Garbage_yolov5s_attn_with_sewage_ar0005_aug_15-02-2022/exp
declare -a TESTS=("test" "test_tong_zhou" "test_xi_an" "test_huan_qiu" "test_tian_an" "test_02" "test_04" "test_06" "test_24" "test_46")

#----------------------
# Evaluation
#----------------------
for i in ${TESTS[@]}; do
    python val.py \
    --data "./data/litter_general.yaml" \
    --test_scene "images/${i}" \
    --weights "./runs/${EXPERIMENT}/exp2/weights/best.pt" \
    --imgsz 512 \
    --conf-thres 0.5 \
    --iou-thres 0.45 \
    --device 2 \
    --save-txt \
    --project "./inference/${EXPERIMENT}" \
    --name ${i} \
    --save-conf
done
