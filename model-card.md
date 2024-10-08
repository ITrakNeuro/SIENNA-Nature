

# SIENNA Model Card

SIENNA is a clinically-focused AI model with fewer than 285K trainable parameters developed to support transferability across tumor types (Glioblastoma Multiforme, Metastatic, Pituatory, Meningioma), MRI modalities (FLAIR, T1ce), and Tesla strengths (1.5T, 3T). SIENNA integrates the PREMO algorithm to enhance image quality and consistency without extensive preprocessing, ensuring strong generalizability to new patient data. This lightweight design makes it suitable for real-world clinical applications while maintaining high diagnostic accuracy.

## Model Details

1.	Development of a minimalistic clinical data pre-processing pipeline (PREMO) to better align ML diagnostic models with real-world clinical scenarios.
2.	Creation of a CNN architecture with 122X less trainable parameters, which is orders of magnitude less compared to 10s of millions that are used typically in state-of-the-art models, that achieves generalizability in ML for clinical diagnostics in a next generation energy efficient footprint.
3.	CNN design focus around optimization of true positive/negative and reduction of false positive/negative outcomes on 2D DICOM images and providing detailed outcomes for multiclass classification tailored for clinical diagnostic decisions.
4.	Demonstration of SIENNA as a generalizable and transferable diagnostic and data handling pipeline adaptable to multiple MRI modalities, such as FLAIR and T1ce, and across expanded tumor class comparisons (Glioblastoma, Pituitary, Meningioma, Metastatic).
5.	Grad-CAM interpretability methodology of SIENNA multiclass classification output predictions.

## Uses

The SIENNA model is intended for clinical deployment in medical imaging diagnostics, with a focus on brain tumor detection and classification. Its lightweight, multilayer CNN architecture requires minimal trainable parameters, making it computationally efficient and suitable for real-time applications in clinical settings. SIENNA addresses common challenges in medical ML models—such as overfitting, lack of generalizability, and high computational costs—by maintaining robust performance across both small and large datasets without overfitting or underfitting.

In brain tumor diagnostics, SIENNA effectively distinguishes between normal physiology and various tumor pathologies like glioblastoma multiforme (GBM) and metastatic tumors. It leverages class-specific spatial features in MRI scans, enhancing diagnostic accuracy as evidenced by Gradient-weighted Class Activation Mapping (Grad-CAM) analyses. By employing a non-interdependent multiclass classification approach and fine-tuning hyperparameters to minimize false positives and negatives, SIENNA achieves high accuracies (90-93%) while reducing the risk of misdiagnosis.

Beyond its immediate application, SIENNA serves as a framework for developing efficient, generalizable AI models in other diagnostic areas. Its design principles promote transferable systems within the medical context, emphasizing the need for computationally less expensive architectures that can handle diverse clinical data effectively.

## Bias, Risks, and Limitations
The SIENNA model, while optimized for high accuracy in brain tumor diagnostics, is trained primarily on datasets that consist largely of Caucasian patients, which may lead to a potential bias in data representation. While the model demonstrates strong performance within its intended scope, it may require additional training on outlier pathologies, ideally with a minimum of 150 samples per condition. At present, the model is focused on brain tumor diagnostics using MRI data, with the potential for expansion into other medical areas as more training data becomes available.

### Recommendations
To improve the SIENNA model’s performance and applicability, training should be expanded to include more diverse patient data across various demographics and medical conditions. Additionally, extending its diagnostic capabilities beyond brain tumors to other pathologies and incorporating multiple imaging modalities, such as CT, will enhance its generalizability and utility in broader clinical contexts.

## Training Details

### Training Data
The de-identified clinical data (Clinical Dataset 1) used in this study for training, validation, and testing of SIENNA were obtained as part of an IRB-approved collaboration with Dr. Pilitsis (Albany Medical College IRB 6127) and includes 17 patients (for complete dataset description see Table 2S). The MRI files encompass Tesla magnetic field strengths of 1.5T and 3T, generated by General Electric SIGNAHDxT 1.5T, GE SIGNA Artist 1.5T, and Philips Ingenia 3T MRI Machines. Thus, our cross-validation assesses SIENNA's generalizability across a patient cohort of males/females ranging in age from 41-70 years for GBM and 30-79 years for MET and encompassing different magnetic strengths and MRI machine manufacturers. Patient ethnicity was primarily Caucasian with one Black patient. All radiographs are axial plane slices and were provided in the Digital Imaging and Communications in Medicine, DICOM, format. The patient files were expert-typed as MET or GBM classes and included multiple 2D image scans (slices) within a 3D Z-plane series. To simplify metadata complexities and reduce image storage demands, we converted individual DICOM files to the Portable Network Graphics (.png) format and the DICOM metadata was re-annotated to adhere to a uniform labeling format for all 17 patients. The format includes information such as patient number, tumor presence (TUM for tumor and NON for non-tumor), tumor type (MET for metastasized and GBM for Glioma), modality type (FLA for FLAIR), and the slice number [P8TUMMETFLA(3)]. These identifiers are utilized post-analysis to re-align data outcomes from individual 2D images to the original patient files to benefit patient-specific analysis.  

### Training Procedure

#### Preprocessing
The SIENNA model applied minimal preprocessing to maintain data integrity and enhance real-world applicability. Unlike common approaches in public brain tumor datasets, steps such as skull stripping were deliberately avoided to keep the model adaptable to diverse clinical settings. Preprocessing focused on resizing MRI scans to 60x60 pixels for efficiency and applying the in-house PREMO (Pixel Redistribution Enhancement Masking Optimization) algorithm to enhance image clarity and standardize pixel intensity. Non-essential Z-slices, such as those from the upper and lower regions of the skull, are trimmed.

#### Training Hyperparameters
The hyperparameter ranges were established through a thorough study of existing literature on optimal parameters for clinical data.

#### Key Model Hyperparameters Dynamically Tuned using Hyperas:
- **Convolutional Layers**: The model consists of three convolutional blocks, with the number of filters per layer dynamically tuned between **10 and 200**, and kernel sizes ranging from **2x2 to 8x8**. **L2 regularization** was applied with a range of **0.00001 to 0.5** to prevent overfitting.
- **Activation Functions**: The layers utilize activations like **ReLU, SELU, ELU, LeakyReLU**, and others, selected based on hyperparameter tuning.
- **Dropout**: Applied after each convolutional block, with rates between **0.1 to 0.5**, preventing overfitting.
- **Dense Layers**: A fully connected layer with **512 neurons** using either **ReLU** or **Sigmoid**, followed by a final **Softmax** classification layer.
- **Batch Size**: Tuned to values of **16, 32, or 64** based on performance and memory utilization.
- **Epochs**: The model was trained for **15 to 25 epochs** with early stopping based on validation loss.

#### Training Regime:
- **Optimizer**: The **Adam optimizer** was used for its adaptive learning rate and convergence speed.
- **Loss Function**: **Categorical cross-entropy** handled the multiclass classification, with additional metrics like **accuracy, false negatives, false positives**, and **recall**.
- **Adversarial Training**: Adversarial examples were used to improve robustness, integrating noise-resistant training.
- **Callbacks**: **Early stopping** and **ModelCheckpoint** were used to monitor validation loss and save the best-performing model.
- **Validation**: The model was evaluated on a test set after each epoch, with the best validation accuracy recorded.
During hyperparameter optimization, **loss, false positives (FP), and false negatives (FN)** were carefully monitored and tuned to ensure their limited occurrence, given the clinical nature of the data and the importance of minimizing diagnostic errors.

## Evaluation

**Evaluation System**

- **Inductive Testing**: SIENNA maintains low training error despite label corruption, showcasing robust performance with noisy data and new unseen patient information.

- **Transductive Testing**: Achieved over 90% average accuracy across three classification tasks, validated through 100 repeated random sub-sampling, indicating stable and consistent performance in handling clinical data. By not setting a random seed during the 100 repeated random sub-sampling, we ensured that each train-test split was different, allowing us to robustly assess the model's stability and consistent performance across varied shuffled datasets.

- **Grad-CAM Analysis**: Applied Grad-CAM to visualize feature importance, with distinctive activation maps. GBM slices show concentrated regions of interest, while MET slices display more spatial spread, reflecting differences in tumor multiplicity and spatial patterns.

- **Label Corruption Data**: Demonstrated lower errors in accuracy and F1 scores compared to deeper architectures as label corruption increased, underscoring the model's resilience to noisy and imperfect data.

- **Comparative Analysis**: Outperformed more complex models in noisy data conditions, avoiding overfitting and ensuring reliable generalization to unseen data without the need for excessive parameter tuning.

- **Visualizations for Explainability**: Provided interpretable visualizations with a focus on the clinical utility of model predictions, ensuring transparency and trust in AI-driven diagnostic support.


### Testing Data, Factors & Metrics

#### Testing Data

**Testing Data**

- **Evaluation on Smaller Dataset (Clinical Dataset 1 - In-House Clinical Dataset)**: SIENNA’s performance was evaluated using a clinical dataset of 17 patients, covering glioblastoma multiforme, metastatic, and non-tumor slices. Evaluate through a 100 repeated random sub-sampling method, SIENNA achieved accuracies of 92% for Non-Tumor (SD=5.5%), 91% for GBM (SD=3.2%), and 93% for MET (SD=2.6%). Detailed assessments of True Positive (TP), True Negative (TN), False Positive (FP), and False Negative (FN), F1 score and AUROC further validated its robust performance.

- **Evaluation on Larger Dataset (Clinical Dataset 2 - Cheng, Jun (2017). brain tumor dataset. figshare. Dataset. https://doi.org/10.6084/m9.figshare.1512427.v5)**: To test SIENNA’s ability to generalize on larger, more heterogeneous datasets, it was evaluated using a clinical brain tumor dataset of 3064 T1-contrast-enhanced images from 233 patients. SIENNA maintained accuracies ranges: 96% for Pituitary Tumors (SD=1.2%), 92.6% for Meningioma (SD=1.6%), and 95.3% for Glioma (SD=1.4%) across 100 random sub-samples. 

- **Evaluation on Higly Processed Public Dataset (BraTS 2020 - Multimodal Brain Tumor Segmentation Challenge (2020).https://www.med.upenn.edu/cbica/brats2020/data.html)**: SIENNA achieved 97% accuracy on highly processed datasets like BraTS for tumor identification.

## Hardware for Training
- **Hardware Type:** NVIDIA GTX 1650 GPU boasting 4 GB of GPU RAM, Colab GPU
- **Hours used for training single model:** ~0.45 hours
- **Cloud Provider:** Google Colab
