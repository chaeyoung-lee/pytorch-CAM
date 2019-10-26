# pytorch-CAM
This repository is an unofficial version of Class Activation Mapping written in PyTorch.

## Class Activation Mapping (CAM)
Paper and Archiecture: [Learning Deep Features for Discriminative Localization][1]

Paper Author Implementation: [metalbubble/CAM][2]

*We propose a technique for generating class activation maps using the global average pooling (GAP) in CNNs. A class activation map for a particular category indicates the discriminative image regions used by the CNN to identify that category. The procedure for generating these maps is illustrated as follows:*

<div align="center">
  <img src="http://cnnlocalization.csail.mit.edu/framework.jpg"><br><br>
</div>

*Class activation maps could be used to intepret the prediction decision made by the CNN. The left image below shows the class activation map of top 5 predictions respectively, you can see that the CNN is triggered by different semantic regions of the image for different predictions. The right image below shows the CNN learns to localize the common visual patterns for the same object class.*

<div align="center">
  <img src="http://cnnlocalization.csail.mit.edu/example.jpg"><br><br>
</div>


## Code Description
**Usage**: `python3 main.py`

**Network**: Inception V3

**Data**: [Kaggle dogs vs. cats][3]
- Download the 'test1.zip' and 'train.zip' files and upzip them.
- Divde the total dataset into train group and test group. As you do that, [images must be arranged in this way][4]:
  ```
  kaggle/train/cat/*.jpg
  kaggle/test/cat/*.jpg
  ```
  
**Checkpoint**
- Checkpoint will be created in the checkpoint folder every ten epoch.
- By setting `RESUME = #`, you can resume from `checkpoint/#.pt`.

  [1]: https://arxiv.org/abs/1512.04150
  [2]: https://github.com/metalbubble/CAM
  [3]: https://www.kaggle.com/c/dogs-vs-cats/data
  [4]: http://pytorch.org/docs/master/torchvision/datasets.html#imagefolder
