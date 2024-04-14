# path to requirements file
export requirements_file="requirements.txt"
# Wandb Params
export WANDB_PROJECT="semantic_segmentation"
# use robotics-lab-diro for wandb entity
export WANDB_ENTITY=""
export WANDB_MODE="online"
export WANDB_API_KEY=""
export WANDB_WATCH="false"
export HYDRA_FULL_ERROR=1
# force compilation on cuda of IABN in place activated batch norm
export IABN_FORCE_CUDA=1
# Training params  
network_name=transformer_bg_voc
# config name residing under experiments folder use der_plus_config for deeplab
config_name=der_transformer_config
initial_increment_all=(15 10 15 19)
increment_all=(1 1 5 1)
training_modes=(overlap disjoint sequential)
num_workers=8
batch_size=12
epochs=30 
ckpt_dir=$SLURM_TMPDIR/continual-semantic-segmentation/checkpoints_back
backbone_weights_path=$SLURM_TMPDIR/backbones
# activating virtual env
echo "Activating virtual env."
# package preparation
module load python/3.8
module load cuda/11.1/cudnn/8.1
echo "Copying Code"
# creates output directory
output_dir="$SLURM_TMPDIR/output"
mkdir -p $SLURM_TMPDIR/output/data
mkdir -p $SLURM_TMPDIR/output/wandb
if test -f "$SLURM_TMPDIR/code"; then
    rm -r -f $SLURM_TMPDIR/code
fi
mkdir -p $SLURM_TMPDIR/code

# copying code from current directory to slurm
cp -r . $SLURM_TMPDIR/code

echo "Installing packages"
# creating virtual env and installing packages
if [ -d "$SLURM_TMPDIR/tmp_venv" ]
then
  echo "Virtual env already exists"
  source $SLURM_TMPDIR/tmp_venv/bin/activate
else
  virtualenv $SLURM_TMPDIR/tmp_venv
  source $SLURM_TMPDIR/tmp_venv/bin/activate
  # add extra packages
  if test -f "$requirements_file"; then
    pip3 install -r $requirements_file
    pip3 install torch==1.10.0+cu111 torchvision==0.11.1+cu111 torchaudio==0.10.0+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
    cd $SLURM_TMPDIR
    # Install IABN
    git clone https://github.com/mapillary/inplace_abn.git
    cd $SLURM_TMPDIR/inplace_abn
    python setup.py install
  fi
fi

cd $SLURM_TMPDIR/code

# experiments
echo "running experiments"
for mode in "${training_modes[@]}"
do
    for i in "${!initial_increment_all[@]}"
    do
        initial_increment=${initial_increment_all[i]}
        increment=${increment_all[i]}
        exp_name=$network_name
        exp_name+=_
        exp_name+=$mode
        exp_name+=$initial_increment
        exp_name+=-
        exp_name+=$increment
        echo "Starting Experiment $exp_name"
        python main.py --config-path conf/experiments --config-name $config_name training.mode=$mode training.num_workers=$num_workers training.name=$exp_name training.initial_increment=$initial_increment training.increment=$increment training.batch_size=$batch_size training.epochs=$epochs training.ckpt_dir=$ckpt_dir network.backbone_weights_path=$backbone_weights_path
        rm -rf $SLURM_TMPDIR/code/checkpoints_back
        rm -rf $SLURM_TMPDIR/code/output_logs
        rm -rf $SLURM_TMPDIR/code/outputs
        rm -rf $SLURM_TMPDIR/code/mem_maps
    done 
done
