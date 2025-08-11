# maskclu
Maskclu


### Requirements
PyTorch >= 1.7.0
python >= 3.7
CUDA >= 9.0
GCC >= 4.9
torchvision



### Maskclu Pre-training
To pretrain MaskClu on ShapeNet training set, run the following command. If you want to try different models or masking ratios etc., first create a new config file, and pass its path to --config.

```
python main.py --config cfgs/pretrain.yaml --exp_name <output_file_name> 
```


### Maskclu Fine-tuning
Fine-tuning on ModelNet40, run:

```
python main.py --config cfgs/finetune_modelnet.yaml \
    --finetune_model --exp_name <output_file_name> --ckpts <path/to/pre-trained/model>
```

### Voting on ModelNet40, run:

```
python main.py --test --config cfgs/finetune_modelnet.yaml \
--exp_name <output_file_name> --ckpts <path/to/best/fine-tuned/model>
```

Fine-tuning few shot on ModelNet40, run:
```
python main.py --way 5 --shot 20 --fold 5 --config cfgs/fewshot.yaml  --finetune_model \
--exp_name fewshot520 --ckpts <path/to/pre-trained/model>
```

Part segmentation on ShapeNetPart, run:
```
python trainpartseg.py --ckpts <path/to/pre-trained/model> --root path/to/data --learning_rate 0.0002 --epoch 300
```