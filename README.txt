DS-HAT: Dual-Stream Hierarchical Attention Transformer for DEM Super-Resolution

This repository contains the official implementation for the paper:

[From Gaps to Granularity: CRPAG-DSHAT based Multi-Modal Deep Learning framework for DEM Void Repair and Super-Resolution reconstruction in Himalayas]
(Insert Publication Link: https://www.google.com/search?q=https://www.sciencedirect.com/science/article/pii/S266739325000201)

The DS-HAT framework is a cutting-edge, dual-stage deep learning pipeline engineered to address critical data limitations—specifically data voids and coarse spatial resolution—in Digital Elevation Models (DEMs) for challenging, high-relief environments like the Himalayan region. This repository is dedicated to the Super-Resolution (SR) stage of the pipeline, which effectively enhances the spatial resolution of globally available low-resolution DEMs, such as the ALOS PALSAR $12.5\text{ m}$ product, to a target grid of $8\text{ m}$. This process involves a $1.5625\times$ Super-Resolution mapping, where the low-resolution ALOS DEM is mapped to a high-resolution target grid, supervised by a pre-processed, high-fidelity reference DEM (HMA DEM). The core objective of this process is to overcome the inherent quality limitations in open-source DEMs, thereby generating super-resolved outputs suitable for sensitive geospatial applications, including detailed hydrological modeling and geomorphological analysis in rugged terrain.

2. Methodology and DS-HAT Architecture

2.1 Dual-Stream Hierarchical Attention Transformer (DS-HAT)

The DS-HAT model is designed as a sophisticated CNN-Transformer hybrid featuring two distinct feature extraction paths to process multi-modal inputs. The Main Stream is dedicated to processing the primary input, the low-resolution ALOS DEM ($1\text{ channel}$), utilizing a stack of Residual-in-Residual Dense Blocks (RRDB). This high-capacity structure is essential for robust feature extraction and accurate principal elevation reconstruction. Concurrently, the Support Stream processes a collection of ten Auxiliary Variables, consisting of topographic, hydrological, and spectral derivatives. This stream uses standard Convolutional Neural Networks (CNNs) augmented with a Squeeze-and-Excitation (SE) Block to perform channel-wise feature re-calibration and enhance the extraction of valuable texture and morphometric information.

2.2 Input Data Modalities and Preparation

The model relies on multi-modal inputs to guide the super-resolution process, with training conducted using $256 \times 256$ pixel patches. The inputs are rigorously pre-processed before entering the network. The Main Input is the raw elevation data ($\text{m}$) from the alos_dem, which is kept in its raw units (no normalization). The Supporting Inputs are crucial and comprise six multi-scale topographic and hydrological derivatives—slope, plan_curv, prof_curv, tang_curv, flow_accu, and str1500—alongside four Sentinel-2 spectral bands (s2_b1, s2_b2, s2_b3, and s2_b4). These auxiliary variables require strict normalization; while some use standard Min-Max scaling, others (like curvature derivatives) use an $\arctan$ transform followed by Min-Max normalization to constrain extreme values, ensuring all support variables are normalized to the range $[0, 1]$. The supervisory Target Label (target), derived from the high-resolution ($8\text{ m}$) reference DEM, is also retained in its raw elevation units ($\text{m}$).

2.3 Attention Fusion Mechanism

The primary innovation of DS-HAT is its Attention Fusion mechanism, which adaptively combines the feature maps extracted from the Main Stream ($F_{\text{main}}$) and the Support Stream ($F_{\text{supp}}$). This mechanism generates a dynamic, pixel-wise attention map ($\alpha$), resulting in the fused feature map ($F_{\text{fused}}$) calculated as: 

$$F_{\text{fused}} = F_{\text{main}} \cdot \alpha + F_{\text{supp}} \cdot (1 - \alpha)$$

 This unique formulation allows the network to dynamically weigh the contribution of the auxiliary data at every pixel location. By controlling the influence of morphometric and spectral features, the mechanism excels at enhancing the reconstruction of fine details, such as sharp ridge lines and subtle stream beds, where the input topographic derivatives are most informative.

2.4 Loss Function and Training

To achieve both high elevation accuracy and geomorphological fidelity, the training utilizes a composite loss function ($L_{\text{total}}$) balancing two distinct components: 

$$L_{\text{total}} = 0.85 \cdot L_{\text{MAE}} + 0.10 \cdot L_{\text{Grad}}$$

 The majority weight is given to the $L_{\text{MAE}}$ (Mean Absolute Error, $\text{L}1$) loss, which enforces fidelity to the true elevation magnitude. Critically, a secondary weight is applied to the Gradient Loss ($L_{\text{Grad}}$), which is calculated as the L1 loss on the 1D spatial gradients of the prediction and the target. By penalizing differences in predicted slopes (first-order derivatives), the Gradient Loss ensures the preservation of local terrain continuity and prevents the generation of artifacts, leading to a topographically consistent final DEM.

3. Setup and Running the Code

3.1 Dependencies and Data Preparation

The framework is implemented using Python and PyTorch. Essential dependencies include torch, rasterio, numpy, and standard scientific libraries. It is highly recommended to use a Python environment configured for GPU acceleration (e.g., CUDA 11.8). Successful execution requires that the multi-modal input data be prepared offline. This data must be saved as a single NumPy archive, processed_patches.npy, which contains all aligned, normalized, and patched input variables as well as the raw target DEM patches. This preparation involves: 1) Sourcing all raw ALOS, Sentinel, and derived rasters; 2) Alignment and Reprojection of all inputs to the exact $8\text{ m}$ target grid; 3) Normalization of the supporting variables to the $[0, 1]$ range; and 4) Patch Extraction ($256 \times 256$ pixels).

3.2 Training and Inference

The deep learning pipeline is run using the DS_HAT_DEM PyTorch module. Training is executed via the train_model function, which handles optimization (Adam), loss calculation, and weight saving (ds_hat_model.pt). Key training parameters to monitor include epochs (typically 150 for convergence) and batch_size (set to 2 in the example due to patch limitations). Post-training, the model is loaded in evaluation mode (model.eval()) for inference on the test set. Evaluation of the super-resolved output is performed by computing quantitative metrics such as MAE, RMSE, and SSIM against the high-resolution target DEM patches.