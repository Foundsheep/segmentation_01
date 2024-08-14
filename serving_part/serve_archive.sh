torch-model-archiver --model-name spr_seg \
--version 1.0 \
--serialized-file ./model_store/model.pt \
--extra-files ./model_store/labelmap.txt,./model_store/label_to_name.json,./model_store/name_to_label.json,../utils/image_utils.py,../configs.py \
--export-path ./model_store \
--handler ts_handler.py \
--config-file ./model_store/hparams.yaml

# --model-file ../training_part/model_loader.py \ # this will force the model initialized, not using model.pt