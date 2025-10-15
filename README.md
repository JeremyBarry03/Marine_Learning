# Benthic Species Recognition Dataset

### **About this event**

The William & Mary AI Club will host its first-ever AI Case Competition during Fall 2025 (Homecoming Weekend). This event is designed as a hybrid between a hackathon and a strategic case competition, where students are asked not only to develop functioning applications but also to present their reasoning, frameworks, and design approaches.

For the Marine & Benthic Science Track, students will develop AI-powered computer vision solutions for automated benthic species identification from underwater survey imagery. Teams will work with real-world seafloor monitoring datasets to build models that can accelerate marine biodiversity assessments, support ecosystem monitoring, and enable efficient analysis of large-scale benthic surveys. The solutions should demonstrate practical applications in marine research, conservation efforts, and long-term ecological monitoring programs.

### **Prompt**

"Design an AI-powered solution for benthic species identification in marine science research and monitoring. The primary task is Image Classification that develop robust classification models using a comprehensive dataset of 10,500 images to accurately identify 7 benthic species (Scallop, Roundfish, Crab, Whelk, Skate, Flatfish, Eel) from single-organism underwater images. As an advanced extension, teams may tackle Object Detection that build detection systems to locate and identify multiple organisms within complex seafloor scenes using a smaller supplementary dataset. The solution should demonstrate how these AI models can ac  
celerate benthic biodiversity assessments, support marine ecosystem monitoring, enable efficient analysis of large-scale underwater surveys, and provide insights into species distribution patterns, population dynamics, and habitat characteristics across temporal and spatial scales."Tasks

### Task 1: Image Classification

Classify benthic organisms into one of 7 species categories from underwater images.

### Task 2: Object Detection

Detect and localize multiple benthic organisms within underwater images using bounding boxes.  
		  
---

## Dataset 1: Classification Dataset

### Structure

classification\_dataset/

├── images/

│   ├── 2015\_image001.jpg

│   ├── 2022\_image002.jpg

│   └── ...

└── labels.txt

**Label Format** (`labels.txt`):

2015\_image001.jpg Scallop

2022\_image002.jpg Eel

### Statistics

| Species | Images | Source Years |
| :---- | :---- | :---- |
| Scallop | 1,500 | 2015, 2022 |
| Roundfish | 1,500 | 2015, 2022 |
| Crab | 1,500 | 2015, 2022 |
| Whelk | 1,500 | 2015, 2022 |
| Skate | 1,500 | 2015, 2022 |
| Flatfish | 1,500 | 2015, 2022 |
| Eel | 1,500 | 2015, 2022 |
| **Total** | **10,500** |  |

---

## Dataset 2: Object Detection Dataset (YOLO Format)

### Structure

detection\_dataset/

├── data.yaml

├── train/

│   ├── images/

│   └── labels/

├── val/

│   ├── images/

│   └── labels/

└── test/

    ├── images/

    └── labels/

**Label Format** (YOLO):

class\_id x\_center y\_center width height

0 0.512 0.458 0.234 0.156

### Statistics

**Dataset Splits**:

| Split | Images | Objects | Avg Objects/Image |
| :---- | :---- | :---- | :---- |
| Train | 1,931 | 2,058 | 1.07 |
| Val | 551 | 575 | 1.04 |
| Test | 277 | 299 | 1.08 |
| **Total** | **2,759** | **2,932** | **1.06** |

**Class Distribution**:

| Class ID | Species | Train | Val | Test | Total |
| :---- | :---- | :---- | :---- | :---- | :---- |
| 0 | Crab | 291 | 89 | 45 | 425 |
| 1 | Eel | 275 | 89 | 43 | 407 |
| 2 | Flatfish | 209 | 62 | 26 | 297 |
| 3 | Roundfish | 287 | 82 | 38 | 407 |
| 4 | Scallop | 419 | 109 | 69 | 597 |
| 5 | Skate | 295 | 76 | 39 | 410 |
| 6 | Whelk | 282 | 68 | 39 | 389 |

