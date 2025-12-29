# Mammography Mass Classification

transfer learning system for classifying mammography masses as benign or malignant. uses resnet50 for feature extraction with downstream random forest classifier on embeddings.

medical imaging project using dicom format mammograms from cbis-ddsm dataset.

## what it does

takes mammography dicom images and classifies breast masses into:
- **benign** (including benign without callback)
- **malignant**

two-stage approach:
1. **resnet50 cnn**: pretrained imagenet weights, frozen base, custom classification head
2. **random forest**: trained on deep feature embeddings from resnet50 penultimate layer

## dataset

cbis-ddsm (curated breast imaging subset of ddsm) mammography dataset:
- training metadata: `mass_case_description_train_set.csv`
- test metadata: `mass_case_description_test_set.csv`
- dicom images with pixel arrays

note: original dataset had file path issue (all ended in `000000.dcm` instead of `1-1.dcm`) which is corrected in preprocessing.

## project structure

```
mammography_classifier/
├── data/                           # csv metadata and dicom files
│   ├── mass_case_description_train_set.csv
│   ├── mass_case_description_test_set.csv
│   └── dicom_files/               # dicom images (not included)
├── models/                         # saved model files
├── logs/                          # training logs
├── visualizations/                # plots and charts
├── config.py                      # configuration loader
├── data_processing.py             # dicom loading and preprocessing
├── models.py                      # resnet50 architecture
├── training.py                    # model training utilities
├── random_forest.py               # ensemble classifier
├── visualization.py               # plotting functions
├── main.py                        # full pipeline execution
└── requirements.txt
```

## setup

1. install dependencies
```bash
pip install -r requirements.txt
```

2. download cbis-ddsm dataset
- place csv files in `./data/`
- place dicom images in `./data/dicom_files/`

3. configure (optional)
```bash
cp .env.example .env
# adjust paths/hyperparameters if needed
```

4. run pipeline
```bash
python main.py
```

## pipeline stages

### 1. data loading
- reads train/test csv metadata
- encodes labels: benign/benign_without_callback → 0, malignant → 1
- corrects file path suffix bug in original dataset

### 2. dicom preprocessing
- loads dicom pixel arrays
- converts grayscale to rgb (resnet requirement)
- resizes to 224x224 with padding (maintains aspect ratio)
- normalizes to 0-1 range using max value

### 3. resnet50 training
- pretrained imagenet weights
- frozen base layers (feature extraction only)
- custom head: global average pooling → dense(128) → dropout(0.5) → sigmoid
- trains for 10 epochs with adam optimizer
- binary crossentropy loss

### 4. embedding extraction
- extracts features from 3rd layer from end (before dropout)
- produces fixed-size embeddings for each image
- used as input for random forest

### 5. random forest classification
- trains on extracted embeddings
- 100 estimators with default params
- provides ensemble approach on deep features
- often matches or beats cnn alone

### 6. evaluation and visualization
- accuracy metrics for both models
- confusion matrix
- training curves (loss/accuracy)
- model comparison chart

## model details

**resnet50 architecture:**
- input: (224, 224, 3) rgb images
- base: resnet50 pretrained on imagenet (frozen)
- custom head: gap → dense(128, relu) → dropout(0.5) → dense(1, sigmoid)
- total params: ~24m (23.5m frozen)

**random forest:**
- trained on 2048-d embeddings (resnet50 output before classification head)
- 100 decision trees
- default sklearn hyperparameters

## typical performance

baseline results on cbis-ddsm test set:
- **resnet50 cnn**: ~75-80% accuracy
- **random forest**: ~78-82% accuracy

random forest often edges out cnn slightly due to ensemble effects on deep features.

## notes and gotchas

- **dicom loading**: pydicom can be slow for large datasets. consider caching preprocessed images
- **memory usage**: full dataset loads all images into memory. batch processing recommended for larger sets
- **class imbalance**: dataset has more benign than malignant cases. consider class weights or resampling
- **file path bug**: original dataset csv had wrong file extensions. preprocessing corrects this automatically
- **rgb conversion**: mammograms are grayscale but resnet expects 3 channels. we duplicate the channel rather than using grayscale-specific models
- **transfer learning**: base frozen means we're using imagenet features. fine-tuning might improve results but risks overfitting on small medical datasets

## future improvements

things to try:
- fine-tune top resnet layers instead of full freeze
- data augmentation (rotations, flips, zoom)
- class weighting for imbalanced dataset
- ensemble resnet + random forest predictions
- gradcam visualizations for interpretability
- cross-validation for more robust metrics
- try efficientnet or vision transformer backbones

## medical disclaimer

this is an educational/research project. not validated for clinical use. any medical imaging classification system requires extensive validation, regulatory approval, and clinical testing before deployment.

## acknowledgments

- cbis-ddsm dataset: lee et al. (2017)
- resnet50: he et al. (2015)
- project for medical imaging course

## license

academic use only - not for clinical deployment
