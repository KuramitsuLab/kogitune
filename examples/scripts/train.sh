#!/bin/bash
#$ -l h_rt=00:03:00
#$-j y
#$-m b
#$-m a
#$-m e
#$-cwd

source /etc/profile.d/modules.sh
module load gcc/12.2.0 python/3.11/3.11.2 cuda/11.7/11.7.1 cudnn/8.8/8.8.1 nccl/2.14/2.14.3-1
export LD_LIBRARY_PATH=/apps/gcc/12.2.0/lib64:/apps/python/3.11.2/lib
source ~/cuda/bin/activate

export PT_CACHE_DIR=$SGE_LOCALDIR

CONFIG_PATH="../config/training_setup.yaml"  # Configuration for the training setup
URLS_PATH="../datasets/urls.txt"             # List of URLs for datasets
MODEL_NAME="Llama2" # available options are 'T5', 'GPT2', 'GPTNeoX', 'Llama2'

python3 train.py --config $CONFIG_PATH --urls $URLS_PATH --model $MODEL_NAME
