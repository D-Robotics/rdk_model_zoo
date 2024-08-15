English| [简体中文](./README_cn.md)

Clip
=======

# 1. Model introduction

CLIP (Contrastive Language-Image Pre-training) is a MultiModal Machine Learning (Text and Image) pre-training model developed by OpenAI. The CLIP model achieves cross-modal understanding by learning how to compare text and images. This contrastive learning method enables CLIP to learn the semantic relationship between text and images without any supervised labels.

The core idea of the CLIP model is to embed text and images into a common semantic space, so that related text descriptions and image content are represented close to each other in this space, while unrelated ones are represented far away. This design enables the CLIP model to perform well in various tasks, such as image classification, image retrieval, text classification, etc.

Characteristics of CLIP model:

1. **MultiModal Machine Learning Embedding** : The CLIP model first embeds text and images into a shared multidimensional space. This space is designed to capture the semantic relationship between text descriptions and image content.

2. **Contrastive Learning** : CLIP uses contrastive learning to train models. In contrastive learning, the model is required to map relevant text descriptions and image content to adjacent locations in space, while irrelevant content is mapped to distant locations. In this way, the model learns how to distinguish between relevant and irrelevant text-image pairs.

3. **Training data** : CLIP is pre-trained using large-scale text and image datasets, where text descriptions and image content are collected from the Internet. These datasets contain a variety of different text descriptions and image content, helping the model learn a wider range of semantic relationships.

4. **Self-supervised learning** : The CLIP model uses the method of self-supervised learning, that is, the model does not need artificial labels during the training process. Instead, the model learns by using the natural association between the text description and the image content in the dataset.

5. **Cross-task applications** : Since CLIP learns the general semantic relationship between text and images, it can be fine-tuned on various tasks, such as image classification, image retrieval, text classification, etc. This versatility makes CLIP perform well in different fields and tasks.

# 2. Model download link

- image encoder: TODO
- text encoder: TODO

img_encoder bin and text_encoder onnx are in the same directory as the current README.md.

# 3. Input and Output Data

## 3.1 Image Encoder

- Input Data

  | Input Data | Data Type | Shape                            | Layout |
  | -------- | -------- | ------------------------------- | ------------ |
  | image    | FLOAT32  | 1 x 3 x 224 x 224 | NCHW           |

- Output Data

  | Input Data | Data Type | Shape                            | Layout |
  | -------- | -------- | ------------------------------- | ------------ |
  | image_feature    | FLOAT32  | 1 x 512 | NCHW           |

## 3.2 Text Encoder 

- Input Data

  | Input Data | Data Type | Shape                            | Layout |
  | -------- | -------- | ------------------------------- | ------------ |
  | texts    | INT32  | num_text x 77 | NCHW           |

- Output Data

  | Input Data | Data Type | Shape                            | Layout |
  | -------- | -------- | ------------------------------- | ------------ |
  | text_features    | FLOAT32  | feature_dim x 512 | NCHW           |