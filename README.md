# Image Classification: Logistic Regression vs. Random Forest

An individual machine-learning project completed for **MATH 3333: Data Analytics — A Hands-on Approach** at York University.

The project compares **logistic regression** and **random forest** for binary image classification. The target is whether an image belongs to the `outdoor-day` category.

## Project overview

The analysis explores two feature representations:

- **Logistic regression:** median red, green, and blue pixel intensities for each image.
- **Random forest:** grayscale image features created by dividing each image into a 10 × 10 grid and calculating the mean intensity in each cell.

Model performance is evaluated using:

- sensitivity
- specificity
- misclassification error
- area under the ROC curve (AUC)

## Results from the submitted course report

| Model | Sensitivity | Specificity | Misclassification Error | AUC |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.502 | 0.874 | 0.255 | 0.807 |
| Random Forest | 0.564 | 0.895 | 0.219 | 0.806 |

In the submitted analysis, random forest produced higher sensitivity and specificity and a lower misclassification error, while logistic regression had a marginally higher AUC.

> **Note:** The code in this repository is a cleaned portfolio refactor of the original course analysis. The table above reports the results documented in the submitted project report, so rerunning this refactored version may not reproduce the exact same values because the train/test split and implementation details have been standardized.

## Repository structure

```text
MATH-3333/
├── image_classification.R
├── README.md
└── .gitignore
```

The image dataset and metadata are intentionally **not redistributed** in this repository.

## Running the analysis

Place the local/course data in:

```text
data/
├── photoMetaData.csv
└── columbiaImages/
    ├── image_1.jpg
    ├── image_2.jpg
    └── ...
```

Install the required R packages if needed:

```r
install.packages(c("jpeg", "randomForest", "pROC"))
```

Then run:

```r
source("image_classification.R")
```

The script validates the input files, extracts features, creates a reproducible stratified 80/20 train/test split, fits both classifiers, and reports sensitivity, specificity, misclassification error, and AUC.

## Tools and methods

**Language:** R

**Libraries:** `jpeg`, `randomForest`, `pROC`

**Methods:** image feature engineering, logistic regression, random forest classification, ROC/AUC analysis, classification metrics

## Data

The submitted report references the **Columbia Photographic Images and Photorealistic Computer Graphics Dataset** (Ng et al., 2005). The underlying image files are not included here because this repository is intended to showcase the analysis rather than redistribute the dataset or course materials.

## Academic integrity

This repository contains a cleaned portfolio version of my individual project code. It does **not** include assignment instructions, lecture notes, instructor-provided course materials, or the original dataset.

## Author

**Samar Shehtou**  
BSc (Honours) Data Science (Health), York University
