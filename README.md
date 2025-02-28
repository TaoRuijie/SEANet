# Audio-Visual Target Speaker Extraction with Selective Auditory Attention (TASLP)

## Code includes:
- Training and evaluation on VoxMix (part of VoxCeleb2)
- Training/evaluation dataset (VoxMix)
- Four architectures: AV-DPRNN, MuSE, AV-Sepformer, SEANet (proposed), including the trained model and training logs.
- More details: [Notes](https://github.com/TaoRuijie/SEANet/blob/main/Notes.md)

## Advantage of this code:

- ***Usage:*** Simple and easy, very few files/codes;
- ***Dataset:*** Provide the extracted lip features. [Link](https://drive.google.com/drive/folders/1i43rQevIRGUd3Ei-E24Voqz5R4TqrxNC?usp=sharing)
- ***Details:*** Share my training logs, models and notes. [Link](https://drive.google.com/drive/folders/1VkQEmYNTjGcWvxeZRw_g6n1kYKMYb0Zq?usp=sharing)

## Dataset prepration

For audio part: prepare the VoxCeleb2 dataset, ***wav*** files.

For visual part: I have uploaded the lip features [here](https://drive.google.com/drive/folders/1i43rQevIRGUd3Ei-E24Voqz5R4TqrxNC?usp=sharing)! You can also generate lip embeddings from the pre-trained [visual encoder](https://github.com/zexupan/MuSE/blob/master/data/voxceleb2-800/3_create_lip_embedding.py) for the mp4 files in VoxMix.

## Training and evaluation

Create your environment, install the dependence (Cuda 12.1 in my experiments)

```
    pip install -r requirements.txt
```

[***Train***] Set the data path in run_train.sh, it has four achitectures:

```
    bash configs/run_train.sh
```

[***Test***] You can set the best model in run_eval.sh for testing. 
The models trained by me can be found [here](https://drive.google.com/drive/folders/1VkQEmYNTjGcWvxeZRw_g6n1kYKMYb0Zq?usp=sharing), include my training/testing logs.

```
    bash configs/run_eval.sh
```

- For multi-GPU training. See [Notes](https://github.com/TaoRuijie/SEANet/blob/main/Notes.md).
- During training, audio mixtures are generated ***online***
- VoxMix dataset: 20K train, 5K val, 3K test samples.

## Performance

Here are the VoxMix results (run again based on this open-source code, slightly different from our paper, 150 epochs for all).

| Method          | Val_SI_SDR | Val_SDR | Test_SI_SDR | Test_SDR |
|-----------------|------------|---------|-------------|----------|
| AV-DPRNN        |   11.44    |  11.94  |   11.12     |  11.56   |
| MuSE            |   10.60    |  11.08  |   10.31     |  10.75   |
| AV-Sepformer    |   12.69    |  13.21  |   12.08     |  12.50   |
| SeaNet (ours)   |   13.42    |  13.91  |   12.95     |  13.39   |

## Reference

We study many useful projects in this work, which includes:

- Lin, Jiuxin, et al. "Av-sepformer: Cross-attention sepformer for audio-visual target speaker extraction." ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2023. [Code](https://github.com/lin9x/AV-Sepformer)
- Pan, Zexu, et al. "USEV: Universal speaker extraction with visual cue." IEEE/ACM Transactions on Audio, Speech, and Language Processing 30 (2022): 3032-3045. [Code](https://github.com/zexupan/MuSE)
- Pan, Zexu, et al. "Muse: Multi-modal target speaker extraction with visual cues." ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2021. [Code](https://github.com/zexupan/USEV)

Thanks for these authors to open source their code!

## Citation
Please cite the following if our paper or code is helpful to your research.
```
@article{tao2025seanet,
  title={Audio-Visual Target Speaker Extraction with Reverse Selective Auditory Attention},
  author={Tao, Ruijie and Qian, Xinyuan and Jiang, Yidi and Li, Junjie and Wang, Jiadong and Li, Haizhou},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP)},
  year={2025}
}
```