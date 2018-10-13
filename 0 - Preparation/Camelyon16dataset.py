# CAMELYON16 Data Set
## Overview
The goal of CAMELYON16 challenge is to evaluate new and existing algorithms for automated detection of metastases in hematoxylin and eosin (H&E) stained whole-slide images of lymph node sections. This task has a high clinical relevance but requires large amounts of reading time from pathologists. Therefore, a successful solution would hold great promise to reduce the workload of the pathologists while at the same time reduce the subjectivity in diagnosis.

For the complete description of the challenge and the data set please visit the [challenge](https://camelyon16.grand-challenge.org) website.

The data set is provided by the [Computational Pathology Group](http://www.diagnijmegen.nl/index.php/Digital_Pathology) of the Radboud University Medical Center in Nijmegen, The Netherlands.

## Data
### Images
The data in this challenge contains a total of 399 whole-slide images (WSIs) of sentinel lymph node from two independent data sets collected in Radboud University Medical Center (Nijmegen, The Netherlands), and the University Medical Center Utrecht (Utrecht, The Netherlands).

The slides are converted to generic [TIFF](https://www.awaresystems.be/imaging/tiff/bigtiff.html) (Tagged Image File Format) using an open-source file converter, part of the [ASAP](https://github.com/GeertLitjens/ASAP) package.

### Annotations
The shared XML files are compatible with the [ASAP](https://github.com/GeertLitjens/ASAP) software. You may download this software and visualize the annotations overlaid on the whole slide image.

The provided XML files may have three groups of annotations ("_0", "_1", or "_2") which can be accessed from the "PartOfGroup" attribute of the Annotation node in the XML file. Annotations belonging to group "_0" and "_1" represent tumor areas and annotations within group "_2" are non-tumor areas which have been cut-out from the original annotations in the first two group.

### Notes about the data
All the images except for the ones mentioned below are fully annotated (all tumor areas have been exhaustively annotated). The annotations for the images listed below are not exhaustive. In other words, there might be tumor areas in these slides which have not been annotated. Most of these slides contain two consecutive sections of the same tissue. In those cases one section is typically exhaustively annotated.:
* tumor_010
* tumor_015
* tumor_018
* tumor_020
* tumor_025
* tumor_029
* tumor_033
* tumor_034
* tumor_044
* tumor_046
* tumor_051
* tumor_054
* tumor_055
* tumor_056
* tumor_067
* tumor_079: Blurred tumor region is not annotated.
* tumor_085
* tumor_092: Blurred tumor region on the adjacent tissue is not annotated.
* tumor_095
* tumor_110

The following files have been intentionally removed from the original data set:
* normal_86: Originally misclassified, renamed to tumor_111.
* test_049: Duplicate slide.

Test set notes:
* test_114: Does not have exhaustive annotations.

### Integrity
The *checksums.txt* file contains the SHA-256 checksums of all the shared CAMELYON16 files. The downloaded files can be checked against this list with *sha256sum*.

### Licensing
See *license.txt* for licensing information.
