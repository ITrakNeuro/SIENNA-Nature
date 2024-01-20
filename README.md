# SIENNA

## Abstract
Biomedical imaging diagnostics and clinical workflow are realizing significant benefits through the integration of Artificial intelligence (AI) and machine learning (ML) to streamline clinical communication and processes, unify expertise, and expand the scope of retrievable information from data. For magnetic resonance imaging (MRI) of brain tumors of glioblastoma (GBM) or metastasized (MET) origins, overlapping and complex tumor features hinder multi-classification. MRI tumor-type misdiagnosis by experts has a significant error rate, according to recent retrospective studies as high as 85%. Multiple diagnostic protocols currently compensate, delaying clinical workups and treatments. ML applied to minimally processed MRI clinical DICOM datasets is largely unexplored for tumor diagnostics, with most studies relying on highly processed public datasets. This hinders clinical integration due to overfitting and low generalizability of the existing ML approaches trained on those datasets and lowers the trustworthiness of such ML models for new patients and less processed data. To enable AI deployment in the clinical setting we developed SIENNA, a deep-learning AI diagnostics framework with state-of-the-art data processing and training of clinical MRI DICOM datasets for neuromorphic analysis. SIENNA cross-validation forces our model to reduce overfitting. Limit false positives and false negatives, and achieve commendable performance for new patients, constituting a real-world clinical scenario. SIENNA demonstrates average accuracy on clinical DICOM MRI data across 3 tasks of 92% (Non-Tumor), 91% (GBM), and 93% (MET), with the distribution of accuracies skewed higher and lower bound at 75%.  SIENNA achieves high accuracy typing of brain tumor GBM and MET pathologies, capability to work with small clinical datasets, and is a computationally light and portable AI companion diagnostic to advance the use of MRI in patient care. 

This repository offers the pre-trained weights of SIENNA, specifically trained on clinical data. The python script retrieves sequentially pre-trimmed MRI scans from designated data paths, applies "Pixel Redistribution Enhancement, Masking, Optimization" (PREMO) for equalization, and makes predictions using the SIENNA pre-trained architecture. The patient-specific results are then presented using metrics that quantify its performance, such as Accuracy and F1-Score.
## Hardware Requirements

The training process was conducted on a Microsoft Windows 11 workstation equipped with an Intel(R) Core (TM) i7-10750H six-core CPU, 16 GB of system RAM, and a single NVIDIA GTX 1650 GPU boasting 4 GB of GPU RAM.

## Python Environment

Python libraries and modules required are listed in requirements.txt.

<pre>pip install -r requirements.txt </pre>


## SIENNA Architecture 

### Source code for testing SIENNA:

1. SIENNA pre-trained.h5: Pre-trained weights of SIENNA on clinical data. 
2. patient_wise_analysis.py: Reads MRI images, applies preprocessing (including the patented PREMO method), loads a pre-trained model, and evaluates its performance on the test data.
