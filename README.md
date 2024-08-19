# segmentation_01
---
### Serving
1. Prerequisites
- Virtual environment is needed
   - `virtualenv .venv -p python3.10`(automatically activates the venv)
   - `pip install -r requirements.txt`
- The serialised model file should be located in `model_store` folder as `model.onnx`
- The model's training-related files should be located in `model_store` to be loaded when `torchserve` starts. Those files are as follows
   - `label_to_name.json`
   - `name_to_label.json`(currently, not used though)
   - `labelmap.txt`
   - `hparams.yaml`
      - This file contains hparams for the to-be-deployed model

2. How to serve
- Follow the instruction below
   1. go to `.../serving_part`
   2. `sh serve_archive.sh`
      - This will make `spr_seg.mar` in `model_store` folder
      - if `spr_seg.mar` already in the folder, it will throw an error to prevent accidental replacement of original model `mar` file
      - if a new model is trained and is going to be deployed, erase the `mar` file first and then, execute `sh serve_archive.sh` command
   3. make sure the preprequisites
      - In total, there will be 6 files in `model_store` folder
   4. `sh serve_run.sh`
   5. `sh serve_stop.sh`(when wanting to stop)

3. Other infomation
<!-- - This serving logic follows `torchserve`'s `script mode` model loading logic
- `eager model` loading logic is executed if `--model-file` argument is passed when invoking `torch-model-archiver`, however this would cause the model to be loaded either from checkpoint, or initialized brand-new.
   - The checkpoint-using loading way is commented in `ts_handler.py`, and uncommenting it and commenting `super().initialize(context)` would change the loading logic to `eager model` with checkpoint -->
   - This serving logic follows `torchserve`s `onnx` model loading logic
      - reference : https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
      - `super().initialize(context)` load a model
   - This serving code assumes to receive one file at an inference time

### Training
1. Prerequisites
- Virtual environment is needed
   - `virtualenv .venv -p python3.10`(automatically activates the venv)
   - `pip install -r requirements.txt`

2. How to train
- Follow the instruction below
   1. go to `.../training_part`
   2. `python train_cli.py --root <dataset_dir_path> --other_arguments`
   3. check the training logs and checkpoint file in `./{timestamp}_{model_name}...` directory
   <!-- 4. `hparams.yaml` and `epock....ckpt` are to be used in serving, so copy them to `../serving_part/model_store`
      - `.ckpt` file should be copied as `model.ckpt` -->
   4. `hparams.yaml` and `model.onnx` are to be used in serving, so copy them to `../serving_part/model_store`