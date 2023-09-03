# DedustNet: A Frequency-dominated Swin Transformer-based network for Agricultural Dust Removal

---

>__Abstract:__ Dust has a significant impact on the environmental perception of automated agricultural machines, and dust removal methods using deep learning approaches still need to be further improved and refined. In this paper, a trainable end-to-end learning network (DedustNet) is specifically proposed to solve the real-world dust removal task. Specially, the frequency-dominated Swin Transformer-based block (DWT-Transformer Block) is proposed to address the limitation of the global receptive field and global information when facing complex dusty background. The Cross-level Information Fusion Module is presented to solve the loss of texture details and color brought by image enhancement algorithms to the processing results. DedustNet trades off the model complexity and the network's dust removal performance with 1.866M, both qualitative and quantitative results show that DedustNet outperforms state-of-the-art methods on reference and non-reference metrics.

* [Network Architecture]()
* [Citation]()
    * [Dependencies and Installation]()
    * [Datasets Preparation]()
        * [Synthetic Fog Dataset]()
        * [Real-world Fog Datasets]()
* [Quick Run]()
* [Inference Time Comparisons]()
* [Qualitative Comparisons(GIF display)]()
* [Ablation experiments]()
    * [Visualization results on the effect of the value of α]()
    * [Visualization comparison of ablation experiments on loss function]()
    * [Visualization comparison of ablation experiments on network structure]()
* [Network Details]()
## Network Architecture

<center>
    <img src='images/network.png'>
</center>

|<img src="images/dwt.png">|<img src="images/cifm.png">|
|:-:|:-:|

## Citation


### Dependencies and Installation

* python3.9
* PyTorch>=1.0
* NVIDIA GPU+CUDA
* numpy
* matplotlib
* pytorch_wavelets
### Datasets Preparation

#### Real-world Dust Dataset:

* Dataset: RB-Dust; Paper: [RESIDE: A Benchmark for Single Image Dehazing](https://arxiv.org/pdf/2306.07244.pdf)(www.agriscapes-dataset.com)

#### Real-world Fog Datasets:

* Dataset: DENSE; Paper: [Dense-Haze: a benchmark for image dehazing with dense-haze and haze-free images](https://arxiv.org/pdf/1904.02904.pdf)
* Dataset: NHHAZE; Paper: [NTIRE 2020 NonHomogeneous Dehazing Challenge (2020)](https://competitions.codalab.org/competitions/22236) ; 
<details>
<summary> FILE STRUCTURE (click to expand) </summary>

```
    DedustNet
    |-- README.md
    |-- datasets
        |--RealWorld
            |-- dust
                |-- hazy
                    |-- *.png
                |-- clear
                    |-- *.png
            |-- dense
                |-- hazy
                    |-- *.png
                |-- clear
                    |-- *.png
            |-- nhhaze
                |-- hazy
                    |-- *.png
                |-- clear
                    |-- *.png
```
</details>

## Quick Run

Trained_models are available at [google drive]() .

*Put  models in the `./trained_models/` folder.*

To test the pre-trained models of Dust, Dense-Haze, and NHHaze on your own images, run:
```shell
python test.py --task Task_Name --input_dir path_to_images
```
Here is an example to perform Dust:

```shell
python test.py --task dust --input_dir dataset/dusttest/hazy
```
We have included some test images from the experimental results presented in our paper at `./image file/` folder. 
* `overhead.py` is a file where you can test network model's parameters (Params), multiply-accumulate operations (MACs), floating-point operations (FLOPs) and inference time.

## Quantitative comparisons

Quantitative comparisons on computational efficiency among DedustNet and SOTA methods, where the floating-point operations and inference time are measured on RGB image with a resolution of 256 × 256.

<center>
    <img src='images/Quantitative comparisons.png'>
</center>

Our method does not have a great advantage in inference time compared to SOTA methods, which is because three DWT and IDWT are used in our network, the process of wavelet transform takes a certain amount of time. However, our method outperforms the SOTS methods in quantitative evaluation metrics (PSNR, SSIM, Entropy, and NIQE) and qualitative comparisons on the RB-Dust dataset. Therefore, our proposed method has a great advantage in a comprehensive view of the number of network parameters, model complexity, and overall network performance.

## Qualitative comparisons

### RB-Dust datasets

<center>
    <img src='images/dustbig.png'>
</center>

|<img src=".\images\dust1.gif">|<img src=".\images\dust2.gif">|<img src=".\images\dust3.gif">|
|:-:|:-:|:-:|

To verify the robustness and effectiveness of DedustNet, we have done extension experiments in fog removal, and as can be seen in Fig.6, DedustNet also achieves satisfactory results in fog removal compared to SOTA methods, demonstrating the robustness and generalization ability of DedustNet.

<center>
    <img src='images/haze.png'>
</center>

#### Dense-Haze

|<img src=".\images\dense1.gif">|<img src=".\images\dense4.gif">|<img src=".\images\dense5.gif">|
|:-:|:-:|:-:|


#### NH-Haze

|<img src=".\images\nh3.gif">|<img src=".\images\nh6.gif">|<img src=".\images\nh8.gif">|
|:-:|:-:|:-:|

## Ablation Study

<center>
    <img src='images/ablationbig.png'>
</center>
