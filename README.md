# Tumor-classifier

The purpose of this code is to solve the following tasks:

A. Binary task
Build a classifier to identify whether there is a tumor in the MRI images.
B. Multiclass task
Build a classifier to identify the type of tumor in each MRI image (meningioma tumor, glioma tumor, pituitary tumor or no tumor). 

Relevant Submission Classes:
1. implementation/BinaryMulticlassSvms.ipynb (Task A and B)
2. implementation/CNN.ipynb (Task B)
Supplementary notes:
- * You must install the data folder in the root directory from: * https://1drv.ms/u/s!AqRPcGlyikGpht4Yjt0JOQ-h9Ne5LA?e=EDpAIC
- The dataset provided has already been transformed into the file data/X_HOG_PCA.pickle to reduce the submission size of this package.
- All other classes other than the ones identified above are included to highlight other areas of research but do not related to the results of the final submission.
- The data has been in some cases reformated and moved into a specific folder structure in order to readily be extracted and batched in tensorflow.
Necessary packages:
See file requirements.txt for all packages
