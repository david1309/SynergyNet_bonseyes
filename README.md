#  <div align="center"> SynergyNet - for Basel Face Model (BFM) and FLAME</div>
3DV 2021: Synergy between 3DMM and 3D Landmarks for Accurate 3D Facial Geometry
Cho-Ying Wu, Qiangeng Xu, Ulrich Neumann, CGIT Lab at University of Souther California

Forked from original [SynergyNet repo](https://github.com/choyingw/SynergyNet) [<a href="https://arxiv.org/abs/2110.09772">paper</a>] [<a href="https://youtu.be/i1Y8U2Z20ko">video</a>] [<a href="https://choyingw.github.io/works/SynergyNet/index.html">project page</a>]

This version is adapted to work for morphable model parameters using the [Basel Face Model](https://faces.dmi.unibas.ch/bfm/) (BFM) and [FLAME model](https://flame.is.tue.mpg.de/). There is one branch for BFM and a separate one for FLAME.

An additional adaptation, is that training relies on using data coming from the output of the Datatool (a Datatool is a tool created by [Bonseyes](https://www.bonseyes.com/) for extracting information from a dataset , [see more info here](https://beta.bonseyes.com//doc/pages/user_guides/datatool_index.html)).



<img src='demo/teaser.png'>


## <div align="center"> Git cloning & Install Requirements</div>
This repository relies in other submodules. Because of this, when cloning the repository please use the command:

`git clone --recurse-submodules https://github.com/david1309/SynergyNet_bonseyes` 

Alternatively use the normal clone command, and then update the submodules:
```bash
git clone https://github.com/david1309/SynergyNet_bonseyes
git submodule init
git submodule update

cd data/datatool_api/
git submodule init
git submodule update
```

Run the following commands to install requirements (the installation assumes your GPU runs in CUDA version 11.3. If your CUDA version is different, go to the Pytorch website and select another version):
```
pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113

pip install -e .
```

After this, place the BFM model *.mat* file inside `bfm_utils/morphable_models/` folder. Download the file from [here](https://drive.google.com/file/d/1V5UAwL8AB_dZoxn4HIUzBDEs4MVRZkPR/view?usp=sharing).



## <div align="center">Training</div>
For easily training the model you can run the provided bash script using the command: `bash train_script.sh`.

For a quick debug to validate changes in the code, run: `bash train_script_debug.sh`.

Important arguments / hyperparemters of the training script are:

* `--datatool-root-dir`: the path to the output folder of the Datatool.
* `--train-tags` / `--val-tags`: names of the datatool tags (subfolder or subdatasets) you want to use for training and validation .
* `--exp-name`: Name of the experiment, used for saving all related checkpoints (saved models, logs, results images, tensorboard files etc.). Checkpoints are saved under `ckpts/<exp-name>`.
* `--debug`: if *True*, quickly runs the training only using few samples of the datatool (few == batch-size). 
* `--epochs, --batch-size, --base-lr`: variables controling training details.

You can also train the model using the command line and passing the desired arguments:

```bash
python main_train.py  --datatool-root-dir="/root/output_debug_all_wv" --train-tags="IBUG" --val-tags="IBUG_Flip" --debug=True --exp-name="debug_cmd" --epochs=10 --batch-size=32
```


**Bibtex**

If you find our work useful, please consider to cite our work 

    @INPROCEEDINGS{wu2021synergy,
      author={Wu, Cho-Ying and Xu, Qiangeng and Neumann, Ulrich},
      booktitle={2021 International Conference on 3D Vision (3DV)}, 
      title={Synergy between 3DMM and 3D Landmarks for Accurate 3D Facial Geometry}, 
      year={2021}
      }


**Acknowledgement**

The project is developed on [<a href="https://github.com/cleardusk/3DDFA">3DDFA</a>] and [<a href="https://github.com/shamangary/FSA-Net">FSA-Net</a>]. Thank them for their wonderful work. Thank [<a href="https://github.com/cleardusk/3DDFA_V2">3DDFA-V2</a>] for the face detector and rendering codes.
