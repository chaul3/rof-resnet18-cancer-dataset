## Methodology

This project investigates the relevance of individual penultimate-layer features in a ResNet-18 model trained for skin lesion classification (ISIC dataset). The methodology is inspired by the approach described in [`zhao24b.pdf`](zhao24b.pdf).

### 1. Baseline Model Evaluation

- The pretrained ResNet-18 model is loaded and evaluated on the test set to establish baseline accuracy metrics.
- The evaluation uses standard classification metrics, including mean accuracy and group-wise accuracies.

### 2. Feature Extraction

- Penultimate-layer (second-to-last layer) features are extracted for all test samples.
- This is done by forwarding each sample through the network up to the average pooling layer, resulting in a feature vector for each image.

### 3. Feature Ranking

- The importance of each penultimate-layer unit is estimated by computing the mean L1-norm (absolute activation) across the test set.
- Units are ranked in descending order of their mean L1-norm, under the hypothesis that higher-activation units are more relevant for classification.

### 4. Relevance of Features (ROF) Analysis

- To assess the contribution of top-ranked units, a masking strategy is applied:
    - For each value of *k* (from 1 up to the total number of units), only the top-*k* units (by L1-norm ranking) are "activated" (retained); all others are set to zero.
    - The model’s forward pass is patched to use these masked features for classification.
- The model is evaluated for each *k*, and accuracy metrics are recorded, resulting in accuracy-vs-units curves.

### 5. Visualization

- The accuracy curves for different metrics (mean accuracy, group accuracies, etc.) are plotted as a function of the number of activated units.
- The baseline accuracy is shown as a reference.
- These plots illustrate how model performance depends on the number and identity of penultimate-layer features, providing insight into feature relevance and redundancy.

For further details and theoretical background, please refer to the origin
