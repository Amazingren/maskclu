<h2>
Masked Clustering Prediction for Unsupervised Point Cloud Pre-trainin (AAAI'26, Oral) [Under Construction]
</h2>


The official implementation of our work: <strong>MaskClu</strong>.

![visitors](https://visitor-badge.laobi.icu/badge?page_id=Amazingren.maskclu)
[![arXiv](https://img.shields.io/badge/arXiv-2503.23367-b31b1b.svg)](https://arxiv.org/pdf/2508.08910)
[![Homepage](https://img.shields.io/badge/Project-Page-orange)](https://arxiv.org/pdf/2508.08910)

#### $^\star$[Bin Ren](https://amazingren.github.io/)<sup>1,2</sup>, $^\star$[Xiaoshui Huang](https://xiaoshuihuang.github.io/)<sup>3</sup>, [Mengyuan Liu](https://scholar.google.com/citations?user=woX_4AcAAAAJ&hl=en)<sup>4</sup>, [Hong Liu](https://scholar.google.com/citations?user=WLMUAjsAAAAJ&hl=zh-CN)<sup>4</sup>, [Fabio Poiesi](https://scholar.google.com/citations?user=BQ7li6AAAAAJ&hl=zh-CN) <sup>5</sup>, [Nicu Sebe](https://scholar.google.com/citations?user=stFCYOAAAAAJ&hl=en)<sup>1</sup>, $^\dagger$[Guofeng Mei](https://scholar.google.com/citations?user=VsmIGqsAAAAJ&hl=en)<sup>5</sup>
$^\star$: Equal Contribution, $^\dagger$: Corresponding Author <br>

<sup>1</sup> University of Trento, Italy <br>
<sup>2</sup> University of Pisa, Italy <br>
<sup>3</sup> Shanghai Jiao Tong University, China <br>
<sup>4</sup> Peking University, China <br>
<sup>5</sup> Fondazione Bruno Kessler, Italy <br>

## 📑 Contents

- [News](#news)
- [TODO](#todo)
- [Results](#results)
- [Citation](#cite)

## <a name="news"></a> 🆕 News

- **2025-12:** We're working on releasing the code!
- **2025-11:** Congrats! Our MaskClu has been accepted by AAAI2026 as Oral😊
- **2025-08:** Our paper is available on arXiv.


## <a name="todo"></a> ☑️ TODO

- [ ] Further improvements
- [x] Release code (Under Construction)
- [x] arXiv version available

</div>

> **Abstract:**  Vision transformers (ViTs) have recently been widely applied to 3D point cloud understanding, with masked autoencoding as the predominant pre-training paradigm. However, the challenge of learning dense and informative semantic features from point clouds via standard ViTs remains underexplored. We propose MaskClu, a novel unsupervised pre-training method for ViTs on 3D point clouds that integrates masked point modeling with clustering-based learning. MaskClu is designed to reconstruct both cluster assignments and cluster centers from masked point clouds, thus encouraging the model to capture dense semantic information. 
Additionally, we introduce a global contrastive learning mechanism that enhances instance-level feature learning by contrasting different masked views of the same point cloud. 
By jointly optimizing these complementary objectives, i.e., dense semantic reconstruction, and instance-level contrastive learning. MaskClu enables ViTs to learn richer and more semantically meaningful representations from 3D point clouds.
We validate the effectiveness of our method via multiple 3D tasks, including part segmentation, semantic segmentation, object detection, and classification, where MaskClu sets new competitive results.
Our code will be released at: https://github.com/Amazingren/maskclu.


⭐If this work is helpful for you, please help star this repo. Thanks!🤗


## Dataset Preparation:
TBD



### Requirements

```
# Step1:
conda create -n pcd python=3.11 -y
conda activate pcd


# Step2: Pytorch & faiss-gpu
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121

conda install -c pytorch faiss-gpu

# Step3: 
pip install -r requirements.txt

# Step4:
pip install https://data.pyg.org/whl/torch-2.1.0%2Bcu121/torch_cluster-1.6.3%2Bpt21cu121-cp311-cp311-linux_x86_64.whl

```




### Maskclu Pre-training
To pretrain MaskClu on ShapeNet training set, run the following command. If you want to try different models or masking ratios etc., first create a new config file, and pass its path to --config.

```
python main.py --config cfgs/pretrain.yaml --exp_name <output_file_name> 
```


### Maskclu Fine-tuning
Fine-tuning on ModelNet40, run:

```
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/finetune_modelnet.yaml \
    --finetune_model --exp_name modelnet40 --ckpts exps/pretrain/cfgs/maskclu_pretrain/ckpt-epoch-300.pth
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


## <a name="cite"></a> 🥰 Citation
Please cite us if our work is useful for your research.

```
@article{ren2025masked,
  title={Masked Clustering Prediction for Unsupervised Point Cloud Pre-training},
  author={Ren, Bin and Huang, Xiaoshui and Liu, Mengyuan and Liu, Hong and Poiesi, Fabio and Sebe, Nicu and Mei, Guofeng},
  journal={arXiv preprint arXiv:2508.08910},
  year={2025}
}
```


## License
This work follows the (MIT License).

## Contact
If you have any questions during your reproduce, feel free to approach me at bin.ren.mondo@gmail.com