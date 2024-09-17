# ScE-GAN
Here we provide an elaboration on the details within the paper "Scene Embedded Generative Adversarial Networks
for Semi-Supervised SAR-to-Optical Image Translation"(GRSL2024) as well as the implementation details of the code.

## Detials：
Due to the limitations of the paper's length, here we elaborate on some detailed information within the paper. This mainly includes the detailed information of the overall framework's modules, the detailed definition of the loss function, and the detailed information of the input dataset. 

**framework of (a)SR block and (b)SFA block**:
Supplementary to Figure 1 of the paper, the detailed expression of the framework's modules is shown in the following. 
we adopt residual block and Transformer to design two blocks that effectively utilize scene category information, i.e., the scene-aware residual (SR) block and the scene fusion attention (SFA) block.The framework of them is shown in folowing Figure (a) and (b).
By substituting the intermediate layer of CycleGAN’s generator with the configuration of SR × 4 + SF A × 1 + SR × 4, we derive the scene information fusion generator (SIFG) inour design.

<img src='imgs/Figure.png' width="800"/>

**detials of loss function**:

