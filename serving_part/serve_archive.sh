torch-model-archiver --model-name spr_seg \
--version 1.0 \
--model-file ../training_part/model_loader.py \
--serialized-file ./model_store/model.ckpt \
--extra-files ./model_store/labelmap.txt,./model_store/label_to_name.json,./model_store/name_to_label.json,../utils/ \
--export-path ./model_store \
--handler ts_handler.py