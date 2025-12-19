DATA_PATH="./data/TUS/test/masks"
EVAL_METRIC_PATH="./output/BCMamba"
MODEL_NAME="BCMamba"
GPU_ID="2,3"

echo "---------------------------------------------start training------------------------------------------------------" &&
accelerate launch \
      --num_processes=3 \
      --num_machines=1 \
      --gpu_ids='0,1,2' \
      --mixed_precision=fp16 \
      --dynamo_backend=no \
      --main_process_port=29536 \
        train.py  &&

echo "----------------------------------------------strat testing-----------------------------------------------------" &&
CUDA_VISIBLE_DEVICES=1 python test.py

#accelerate launch --num_processes=4 --num_machines=1 --gpu_ids=0,1,2,3 --mixed_precision=fp16 --dynamo_backend=no --main_process_port=29536  train.py

#echo "Computing dice..."
#python evaluate/endoscopy_DSC_Eval.py \
#    --gt_path "${DATA_PATH}" \
#    --seg_path "${EVAL_METRIC_PATH}/ph1" \
#    --save_path "${EVAL_METRIC_PATH}/metric_DSC.csv"  &&
#
#echo "Computing NSD..."
#python evaluate/endoscopy_NSD_Eval.py \
#    --gt_path "${DATA_PATH}" \
#    --seg_path "${EVAL_METRIC_PATH}/ph1" \
#    --save_path "${EVAL_METRIC_PATH}/metric_NSD.csv" &&
##
echo "Done."
#rm -rf $0