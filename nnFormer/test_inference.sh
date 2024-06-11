#!/bin/bash


while getopts 'c:n:t:r:p' OPT; do
    case $OPT in
        c) cuda=$OPTARG;;
        n) name=$OPTARG;;
        t) task=$OPTARG;;
        r) train="false";;
        p) predict="true";;
        
    esac
done
echo $name	
fold="nnformer_tumor"

	cd /home/ubuntu/Desktop/nnFormer/nnformer/DATASET/test
	CUDA_VISIBLE_DEVICES=${cuda} nnFormer_predict -i imagesTs -o inferTs/${name} -m 3d_fullres -t ${task} -f 0 -chk model_best -tr nnFormerTrainerV2_${name}
	python /home/ubuntu/Desktop/nnFormer/nnformer/inference_tumor.py $fold




