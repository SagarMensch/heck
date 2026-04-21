---
license: other
license_name: nvidia-open-model-license
license_link: >-
  https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
language:
- en
pipeline_tag: object-detection
arxiv: None
tags:
- image
- detection
- pdf
- ingestion
- yolox
---
# Nemotron Page Element v3

## Model Overview

![viz.png](viz.png)
*Preview of the model output on the example image.*

### Description

The **Nemotron Page Elements v3** model is a specialized object detection model designed to identify and extract elements from document pages. While the underlying technology builds upon work from [Megvii Technology](https://github.com/Megvii-BaseDetection/YOLOX), we developed our own base model through complete retraining rather than using pre-trained weights. YOLOX is an anchor-free version of YOLO (You Only Look Once), this model combines a simpler architecture with enhanced performance. The model is trained to detect **tables**, **charts**, **infographics**, **titles**, **header/footers** and **texts** in documents.

This model supersedes the [nemotron-page-elements](https://build.nvidia.com/nvidia/nemoretriever-page-elements-v2) model and is a part of the NVIDIA Nemotron family of NIM microservices specifically for object detection and multimodal extraction of enterprise documents.

This model is ready for commercial/non-commercial use. 

We are excited to announce the open sourcing of this commercial model. For users interested in deploying this model in production environments, it is also available via the model API in NVIDIA Inference Microservices (NIM) at [nemoretriever-page-elements-v2](https://build.nvidia.com/nvidia/nemoretriever-page-elements-v2).

### License/Terms of use

The use of this model is governed by the [NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) and the use of the post-processing scripts are licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0.txt).

### Team

- Theo Viel
- Bo Liu
- Darragh Hanley
- Even Oldridge

Correspondence to Theo Viel (tviel@nvidia.com) and Bo Liu (boli@nvidia.com)

### Deployment Geography

Global

### Use Case

The **Nemotron Page Elements v3** model  is designed for automating extraction of text, charts, tables, infographics etc in enterprise documents. It can be used for document analysis, understanding and processing. Key applications include:
- Enterprise document extraction, embedding and indexing
- Augmenting Retrieval Augmented Generation (RAG) workflows with multimodal retrieval
- Data extraction from legacy documents and reports

### Release Date

10/23/2025 via https://huggingface.co/nvidia/nemotron-page-elements-v3

### References

- YOLOX paper: https://arxiv.org/abs/2107.08430
- YOLOX repo: https://github.com/Megvii-BaseDetection/YOLOX
- Previous version of the Page Element model: https://build.nvidia.com/nvidia/nemoretriever-page-elements-v2
- Technical blog: https://developer.nvidia.com/blog/approaches-to-pdf-data-extraction-for-information-retrieval/

### Model Architecture

**Architecture Type**: YOLOX <br>
**Network Architecture**: DarkNet53 Backbone \+ FPN Decoupled head (one 1x1 convolution \+ 2 parallel 3x3 convolutions (one for the classification and one for the bounding box prediction). YOLOX is a single-stage object detector that improves on Yolo-v3. <br>
**This model was developed based on the Yolo architecture** <br>
**Number of model parameters**: 5.4e7 <br>

### Input

**Input Type(s)**: Image <br>
**Input Format(s)**: Red, Green, Blue (RGB) <br>
**Input Parameters**: Two-Dimensional (2D)<br>
**Other Properties Related to Input**: Image size resized to `(1024, 1024)`

### Output

**Output Type(s)**: Array <br>
**Output Format**: A dictionary of dictionaries containing `np.ndarray` objects. The outer dictionary has entries for each sample (page), and the inner dictionary contains a list of dictionaries, each with a bounding box (`np.ndarray`), class label, and confidence score for that page. <br>
**Output Parameters**: One-Dimensional (1D) <br>
**Other Properties Related to Output**: The output contains bounding boxes, detection confidence scores, and object classes (chart, table, infographic, title, text, headers and footers). The thresholds used for non-maximum suppression are `conf_thresh=0.01` and `iou_thresh=0.5`. <br>
**Output Classes**: <br>
  * Table
    * Data structured in rows and columns
  * Chart
    * Specifically bar charts, line charts, or pie charts
  * Infographic
    * Visual representations of information that is more complex than a chart, including diagrams and flowcharts
    * Maps are _not_ considered infographics
  * Title
    * Titles can be section titles, or table/chart/infographic titles
  * Header/footer
    * Page headers and footers
  * Text
    * Texts are regions of one or more text paragraphs, or standalone text not belonging to any of the classes above

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA’s hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions.

### Usage

The model requires torch, and the custom code available in this repository.

1. Clone the repository

- Make sure git-lfs is installed (https://git-lfs.com)
```
git lfs install
```
- Using https
```
git clone https://huggingface.co/nvidia/nemotron-page-elements-v3
```
- Or using ssh
```
git clone git@hf.co:nvidia/nemotron-page-elements-v3
```
Optional:
This can be installed as a package using pip
```
cd nemotron-page-elements-v3
pip install -e .
```
2. Run the model using the following code:

```
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from nemotron_page_elements_v3.model import define_model
from nemotron_page_elements_v3.utils import plot_sample, postprocess_preds_page_element, reformat_for_plotting

# Load image
path = "./example.png"
img = Image.open(path).convert("RGB")
img = np.array(img)

# Load model
model = define_model("page_element_v3")

# Inference
with torch.inference_mode():
    x = model.preprocess(img)
    preds = model(x, img.shape)[0]

print(preds)

# Post-processing
boxes, labels, scores = postprocess_preds_page_element(preds, model.thresholds_per_class, model.labels)

# Plot
boxes_plot, confs = reformat_for_plotting(boxes, labels, scores, img.shape, model.num_classes)

plt.figure(figsize=(15, 10))
plot_sample(img, boxes_plot, confs, labels=model.labels)
plt.show()
```

Note that this repository only provides minimal code to infer the model.
If you wish to do additional training, [refer to the original repo](https://github.com/Megvii-BaseDetection/YOLOX).

3. Advanced post-processing

Additional post-processing might be required to use the model as part of a data extraction pipeline. 
We provide examples in the notebook `Demo.ipynb`.

<!---
### Software Integration

**Runtime Engine(s):**
- **Nemotron Page Elements v3** NIM

**Supported Hardware Microarchitecture Compatibility [List in Alphabetic Order]:**
- NVIDIA Ampere
- NVIDIA Hopper
- NVIDIA Lovelace

**Preferred/Supported Operating System(s):**
- Linux

The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment.
This AI model can be embedded as an Application Programming Interface (API) call into the software environment described above.
--->

## Model Version(s):

* `nemotron-page-elements-v3`

## Training and Evaluation Datasets:

### Training Dataset

**Data Modality**: Image <br>
**Image Training Data Size**: Less than a Million Images <br>
**Data collection method by dataset**: Automated <br>
**Labeling method by dataset**: Hybrid: Automated, Human <br>
**Pretraining (by NVIDIA)**: 118,287 images of the [COCO train2017](https://cocodataset.org/#download) dataset <br>
**Finetuning (by NVIDIA)**: 36,093 images from [Digital Corpora dataset](https://digitalcorpora.org/), with annotations from [Azure AI Document Intelligence](https://azure.microsoft.com/en-us/products/ai-services/ai-document-intelligence) and data annotation team <br>
**Number of bounding boxes per class**: 35,328 tables, 44,178 titles, 11,313 charts and 6,500 infographics, 90,812 texts and 10,743 header/footers. The layout model of Document Intelligence was used with `2024-02-29-preview` API version.

### Evaluation Dataset

The primary evaluation set is a cut of the Azure labels and digital corpora images. Number of bounding boxes per class: 1,985 tables, 2,922 titles, 498 charts, 572 infographics, 4,400 texts and 492 header/footers. Mean Average Precision (mAP) was used as an evaluation metric, which measures the model's ability to correctly identify and localize objects across different confidence thresholds.

**Data collection method by dataset**: Hybrid: Automated, Human <br>
**Labeling method by dataset**: Hybrid: Automated, Human <br>
**Properties**: We evaluated with Azure labels from manually selected pages, as well as manual inspection on public PDFs and powerpoint slides.

**Per-class Performance Metrics**:
| Class       | AP (%) | AR (%) |
|:------------|:-------|:-------|
| table       | 44.643 | 62.242 |
| chart       | 54.191 | 77.557 |
| title       | 38.529 | 56.315 |
| infographic | 66.863 | 69.306 |
| text        | 45.418 | 73.017 |
| header_footer | 53.895 | 75.670 |


<!---
## Inference:

**Acceleartion Engine**: TensorRT <br>
**Test hardware**: See [Support Matrix from NIM documentation](https://docs.nvidia.com/nim/ingestion/object-detection/latest/support-matrix.html#)
--->

## Ethical Considerations

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.


## Bias

| Field | Response |
| ----- | ----- |
| Participation considerations from adversely impacted groups [protected classes](https://www.senate.ca.gov/content/protected-classes) in model design and testing | None |
| Measures taken to mitigate against unwanted bias | None |

## Explainability

| Field | Response |
| ----- | ----- |
| Intended Task/Domain: | Document Understanding |
| Model Type: | YOLOX Object Detection for Charts, Tables, Infographics, Header/footers, Texts, and Titles |
| Intended User: | Enterprise developers, data scientists, and other technical users who need to extract structural elements from documents. |
| Output: | After post-processing, the output is three numpy array that contains the detections: `boxes [N x 4]` (format is normalized `(x_min, y_min, x_max, y_max)`), associated classes: `labels [N]` and confidence scores: `scores [N]`.|
| Describe how the model works: | The model identifies objects in an image by first dividing the image into a grid. For each grid cell, it extracts visual features and simultaneously predicts which objects are present (for example, 'chart' or 'table') and where they are located in that cell, all in a single pass through the image. |
| Name the adversely impacted groups this has been tested to deliver comparable outcomes regardless of: | Not Applicable |
| Technical Limitations & Mitigation: | The model may not generalize to unknown document types/formats not commonly found on the web. Further fine-tuning might be required for such documents. |
| Verified to have met prescribed NVIDIA quality standards: | Yes |
| Performance Metrics: | Mean Average Precision, detectionr recall and visual inspection |
| Potential Known Risks: | This model may not always detect all elements in a document. |
| Licensing & Terms of Use: | Use of this model is governed by [NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) and the use of the post-processing scripts are licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0.txt). |

## Privacy

| Field | Response |
| ----- | ----- |
| Generatable or reverse engineerable personal data? | No |
| Personal data used to create this model? | No |
| Was consent obtained for any personal data used? | Not Applicable |
| How often is the dataset reviewed? | Before Release |
| Is there provenance for all datasets used in training? | Yes |
| Does data labeling (annotation, metadata) comply with privacy laws? | Yes |
| Is data compliant with data subject requests for data correction or removal, if such a request was made? |  No, not possible with externally-sourced data. |
| Applicable Privacy Policy | https://www.nvidia.com/en-us/about-nvidia/privacy-policy/ |

## Safety

| Field | Response |
| ----- | ----- |
| Model Application Field(s): | Object Detection for Retrieval, focused on Enterprise |
| Describe the life critical impact (if present). | Not Applicable |
| Use Case Restrictions: | Abide by [NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) and the use of the post-processing scripts are licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0.txt).   |
| Model and dataset restrictions: | The Principle of least privilege (PoLP) is applied limiting access for dataset generation and model development. Restrictions enforce dataset access during training, and dataset license constraints adhered to. |
