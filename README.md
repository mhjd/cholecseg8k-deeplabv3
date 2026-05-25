# README

Train deeplabv3 on cholecseg8k dataset.

## Dataset

Download CholecSeg8k manually on [Hugging Face](https://huggingface.co/datasets/minwoosun/CholecSeg8k/blob/main/data/CholecSeg8k.zip) and place the dezipped dataset in the `dataset/` folder (that you should)


### Dataset Mask Encoding Notes

For each image, the dataset provides a color mask for visualization and a class mask, called the watershed mask, intended for programmatic use.

One important detail is that the watershed masks do not store the foreground classes as contiguous IDs from 1 to 12. Instead, they use non-contiguous grayscale RGB codes such as `(11, 11, 11)`, `(21, 21, 21)`, and `(31, 31, 31)`. The class-to-code mapping is provided in the [CholecSeg8k Kaggle dataset description](https://www.kaggle.com/datasets/newslab/cholecseg8k), where each semantic class is associated with its watershed mask RGB code. These raw codes are mapped to contiguous class IDs `0-12` before training.

An issue is the presence of pixels with value `(255, 255, 255)` in almost all watershed masks. The dataset description does not clearly explain the semantic meaning of this value.  Manual inspection suggests that `(255, 255, 255)` pixels in the watershed masks mainly come from the white border around the endoscopic field of view, and more occasionally from boundaries between annotated regions. Because the dataset documentation does not clearly define the meaning of this value, I will treat it as an ignore label during training.

A second edge case is the rare presence of the raw code `(0, 0, 0)` in a small number of watershed masks. This value is not part of the documented class-to-code mapping. Given its extremely low frequency, I will treat it as an ignore label rather than as a semantic class, while acknowledging that its exact meaning is not clearly documented