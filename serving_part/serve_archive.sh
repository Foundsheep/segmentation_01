torch-model-archiver --model-name spr_seg \
--version 1.0 \
--model-file ../training_part/model_loader.py \
--serialized-file ./model_store/model.ckpt \
--extra-files ./model_store/labelmap.txt, label_to_name.json, name_to_label.json \
--handler ts_handler.py