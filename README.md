# Modification-2 of HD-CapsNet

Hierarchical Deep Capsule Network For Image Classification. The model employs capsule network for each hierarchical levels, treating as a global classification model.

<!-- **The architecture of the BUH-CapsNet with Consistency loss is as follows:**
 
![fig:Architecture](Results/Architecture_BUH_CapsNet.png?raw=true "Architecture of the BUH-CapsNet") -->

**The architecture of the HD-CapsNet is as follows:**
![fig:Architecture](Results/Architecture_HD_CapsNet.png?raw=true "Architecture of the HD-CapsNet")

# Results:

---

| Dataset       | Models           | Model Description                              | Total  Trainable``params (M) | Accuracy Coarse | Accuracy Medium | Accuracy Fine | Hierarchical Precision | Hierarchical Recall | Hierarchical F1-Score | Consistency | Exact Match |
| ------------- | ---------------- | ---------------------------------------------- | ---------------------------- | --------------- | --------------- | ------------- | ---------------------- | ------------------- | --------------------- | ----------- | ----------- |
|               |                  |                                                |                              |                 |                 |               |                        |                     |                       |             |             |
| Fashion MNIST | HD_CapsNet       | HD-CapsNet Model                               | 4.82                         | 99.92%          | 97.79%          | 94.83%        | 97.51%                 | 97.54%              | 97.52%                | 99.84%      | 94.74%      |
| Fashion MNIST | HD_CapsNet_WO-Lc | HD-CapsNet Model Without Consistency Loss (Lc) | 4.82                         | 99.89%          | 97.78%          | 94.92%        | 97.53%                 | 97.59%              | 97.55%                | 99.70%      | 94.77%      |
| Fashion MNIST | HD_CapsNet_WO-SC | HD-CapsNet Model Without Skip Connection       | 4.73                         | 99.91%          | 97.63%          | 94.66%        | 97.40%                 | 97.42%              | 97.41%                | 99.87%      | 94.60%      |
|               |                  |                                                |                              |                 |                 |               |                        |                     |                       |             |             |
| CIFAR-10      | HD_CapsNet       | HD-CapsNet Model                               | 5.23                         | 98.79%          | 94.28%          | 91.22%        | 94.74%                 | 94.89%              | 94.80%                | 99.18%      | 90.95%      |
| CIFAR-10      | HD_CapsNet_WO-Lc | HD-CapsNet Model Without Consistency Loss (Lc) | 5.23                         | 98.71%          | 94.01%          | 90.97%        | 94.53%                 | 94.73%              | 94.62%                | 98.99%      | 90.58%      |
| CIFAR-10      | HD_CapsNet_WO-SC | HD-CapsNet Model Without Skip Connection       | 4.84                         | 98.76%          | 93.36%          | 90.26%        | 94.09%                 | 94.30%              | 94.18%                | 98.94%      | 89.85%      |
|               |                  |                                                |                              |                 |                 |               |                        |                     |                       |             |             |
| CIFAR-100     | HD_CapsNet       | HD-CapsNet Model                               | 7.85                         | 86.93%          | 79.31%          | 66.38%        | 77.43%                 | 79.20%              | 78.12%                | 89.80%      | 63.80%      |
| CIFAR-100     | HD_CapsNet_WO-Lc | HD-CapsNet Model Without Consistency Loss (Lc) | 7.85                         | 86.81%          | 78.73%          | 66.23%        | 77.10%                 | 79.02%              | 77.85%                | 88.62%      | 63.36%      |
| CIFAR-100     | HD_CapsNet_WO-SC | HD-CapsNet Model Without Skip Connection       | 5.55                         | 86.57%          | 78.33%          | 57.08%        | 73.86%                 | 75.00%              | 74.31%                | 92.51%      | 56.10%      |
|               |                  |                                                |                              |                 |                 |               |                        |                     |                       |             |             |
| Marine Tree   | HD_CapsNet       | HD-CapsNet Model                               | 13.58                        | 89.88%          | 78.60%          | 57.15%        | 75.02%                 | 76.04%              | 75.44%                | 94.47%      | 55.59%      |
| Marine Tree   | HD_CapsNet_WO-Lc | HD-CapsNet Model Without Consistency Loss (Lc) | 13.58                        | 89.50%          | 77.57%          | 53.75%        | 73.29%                 | 74.76%              | 73.88%                | 92.37%      | 51.85%      |
| Marine Tree   | HD_CapsNet_WO-SC | HD-CapsNet Model Without Skip Connection       | 5.97                         | 86.98%          | 77.82%          | 55.04%        | 73.35%                 | 75.76%              | 74.36%                | 86.95%      | 49.34%      |
|               |                  |                                                |                              |                 |                 |               |                        |                     |                       |             |             |
| CU Bird       | HD_CapsNet       | HD-CapsNet Model                               | 106.01                       | 40.42%          | 21.61%          | 13.39%        | 23.47%                 | 30.33%              | 26.01%                | 27.34%      | 8.63%       |
| CU Bird       | HD_CapsNet_WO-Lc | HD-CapsNet Model Without Consistency Loss (Lc) | 106.01                       | 36.59%          | 17.78%          | 10.87%        | 20.29%                 | 26.56%              | 22.62%                | 24.09%      | 6.28%       |
| CU Bird       | HD_CapsNet_WO-SC | HD-CapsNet Model Without Skip Connection       | 47.56                        | 35.66%          | 16.98%          | 2.14%         | 14.97%                 | 20.86%              | 17.13%                | 21.44%      | 1.55%       |
|               |                  |                                                |                              |                 |                 |               |                        |                     |                       |             |             |
| Stanford Cars | HD_CapsNet       | HD-CapsNet Model                               | 81.17                        | 53.34%          | 19.52%          | 14.05%        | 26.73%                 | 34.69%              | 29.73%                | 29.15%      | 8.13%       |
| Stanford Cars | HD_CapsNet_WO-Lc | HD-CapsNet Model Without Consistency Loss (Lc) | 81.17                        | 47.50%          | 16.39%          | 11.74%        | 23.56%                 | 31.40%              | 26.50%                | 25.76%      | 6.19%       |
| Stanford Cars | HD_CapsNet_WO-SC | HD-CapsNet Model Without Skip Connection       | 25.85                        | 46.01%          | 12.29%          | 1.57%         | 17.10%                 | 24.04%              | 19.79%                | 13.60%      | 0.87%       |

# Analysis:

1. From Modification-1:
   - Slowly increasing the dimension from *Coarse-to-Fine* \[8D>12D>16D (Coarse>Medium>FINE)\]for decreasing the dimension from *Fine-to-Coarse* \[16D>12D>8D (Fine>Medium>Coarse)\] seems to have effects on Mod-1.1 and Mod-1.3.

     - i.e.  **\[16D>12D>8D (Fine>Medium>Coarse)\]** Bottom-UP approach.
   - It seems that, Top-Down approach does not improve with this technique. In fact it drops the model performance.
2. Form Modification-2: We want to use Top-Down approach in the architecture as it is a natural process.
   - Using the top down approach with changing Primary capsule dimension:
     - **Increasing or Decreasing primary capsule did not worked.** Keep P=8D.
   - Increasing dimension in secondary capsules improved model performance (8D>12D>16D vs 32D>16D>8D)
3. Skip Connections with Top-Down approach.
   - Skip connection improved overall model performance for CIFER-10.
   - Skip connection is slightly below in hierarchical performance of bottom-up approach on CIFER-100 Dataset.
     - With consistency loss it performed better then bottom-up approach with out consistency loss.
