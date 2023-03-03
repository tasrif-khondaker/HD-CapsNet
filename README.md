# Modification-2 of HD-CapsNet
Hierarchical Deep Capsule Network For Image Classification. The model employs capsule network for each hierarchical levels, treating as a global classification model.
 
**The architecture of the BUH-CapsNet woth Consistency loss is as follows:**
 
![fig:Architecture](Results/Architecture_BUH_CapsNet.png?raw=true "Architecture of the BUH-CapsNet")

**The architecture of the HD-CapsNet is as follows:**
![fig:Architecture](Results/Architecture_HD_CapsNet.png?raw=true "Architecture of the HD-CapsNet")
 
## The following changes are made in the architecture.

| Modifications |             Approach             | Dimention of primary capsule <br   />($P$) |                                                                                           Dimention of <br />Secondary capsule   <br />($S_{i}$)                                                                                          |                            Loss Function                            |
|:-------------:|:--------------------------------:|:------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------:|
|    Mod-0.1    | Bottom-UP <br />[Fine-to-Coarse] |                     8D                     |                                                                                                   16D>16D>16D <br />(Fine>Medium>Coarse)                                                                                                   |   Hinge loss (Margin Loss) **With** <br />Consistency ($L_{cons}$)  |
|    Mod-0.2    | Bottom-UP <br />[Fine-to-Coarse] |                     8D                     |                                                                                                   16D>16D>16D <br />(Fine>Medium>Coarse)                                                                                                   |   Hinge loss (Margin Loss) **With** <br />Consistency ($L_{cons}$)  |
|    Mod-1.1    | Bottom-UP <br />[Fine-to-Coarse] |                     8D                     |                                                                                                   16D>12D>8D <br />(Fine>Medium>Coarse)                                                                                                   |   Hinge loss (Margin Loss) **With** <br />Consistency ($L_{cons}$)  |
|    Mod-1.2    |  Top-Down <br />[Coarse-to-Fine] |                     8D                     |                                                                                                   8D>12D>16D <br />(Coarse>Medium>FINE)                                                                                                   |   Hinge loss (Margin Loss) **With** <br />Consistency ($L_{cons}$)  |
|    Mod-1.3    | Bottom-UP <br />[Fine-to-Coarse] |                     8D                     |                                                                                                   16D>12D>8D <br />(Fine>Medium>Coarse)                                                                                                   | Hinge loss (Margin Loss) **Without** <br />Consistency ($L_{cons}$) |
|    Mod-1.4    |  Top-Down <br />[Coarse-to-Fine] |                     8D                     |                                                                                                   8D>12D>16D <br />(Coarse>Medium>FINE)                                                                                                   | Hinge loss (Margin Loss) **Without** <br />Consistency ($L_{cons}$) |
|    Mod-2.1    |  Top-Down <br />[Coarse-to-Fine] |                     8D                     |                                                                                                   32D>16D>8D <br />(Coarse>Medium>FINE)                                                                                                   |   Hinge loss (Margin Loss) **With** <br />Consistency ($L_{cons}$)  |
|    Mod-2.2    |  Top-Down <br />[Coarse-to-Fine] |                     16D                    |                                                                                                   32D>16D>8D <br />(Coarse>Medium>FINE)                                                                                                   |   Hinge loss (Margin Loss) **With** <br />Consistency ($L_{cons}$)  |
|    Mod-2.3    |  Top-Down <br />[Coarse-to-Fine] |                     8D                     |                                                                                                   32D>16D>8D <br />(Coarse>Medium>FINE)                                                                                                   | Hinge loss (Margin Loss) **Without** <br />Consistency ($L_{cons}$) |
|    Mod-2.4    |  Top-Down <br />[Coarse-to-Fine] |                     16D                    |                                                                                                   32D>16D>8D <br />(Coarse>Medium>FINE)                                                                                                   | Hinge loss (Margin Loss) **Without** <br />Consistency ($L_{cons}$) |
|    Mod-2.5    |  Top-Down <br />[Coarse-to-Fine] |                     4D                     |                                                                                                   32D>16D>8D <br />(Coarse>Medium>FINE)                                                                                                   |   Hinge loss (Margin Loss) **With** <br />Consistency ($L_{cons}$)  |
|    Mod-2.6    |  Top-Down <br />[Coarse-to-Fine] |                     4D                     |                                                                                                   32D>16D>8D <br />(Coarse>Medium>FINE)                                                                                                   | Hinge loss (Margin Loss) **Without** <br />Consistency ($L_{cons}$) |
|    Mod-2.7    |  Top-Down <br />[Coarse-to-Fine] |                     8D                     |                                                                                                   32D>24D>16D <br />(Coarse>Medium>FINE)                                                                                                  |   Hinge loss (Margin Loss) **With** <br />Consistency ($L_{cons}$)  |
|    Mod-2.8    |  Top-Down <br />[Coarse-to-Fine] |                     8D                     |                                                                                                   32D>24D>16D <br />(Coarse>Medium>FINE)                                                                                                  | Hinge loss (Margin Loss) **Without** <br />Consistency ($L_{cons}$) |
|    Mod-3.1    |  Top-Down <br />[Coarse-to-Fine] |                     8D                     | 32D>16D>8D <br />(Coarse>Medium>FINE) <br />use   skip connections between Secondary Capsules <br   />$Concatenate([P_{caps}, S_{coarse}])$ > input for $S_{medium}$ <br   />$Concatenate([P_{caps}, S_{medium}])$ > input for $S_{fine}$ |   Hinge loss (Margin Loss) **With** <br />Consistency ($L_{cons}$)  |
|    Mod-3.2    |  Top-Down <br />[Coarse-to-Fine] |                     8D                     | 32D>16D>8D <br />(Coarse>Medium>FINE) <br />use   skip connections between Secondary Capsules <br   />$Concatenate([P_{caps}, S_{coarse}])$ > input for $S_{medium}$ <br   />$Concatenate([P_{caps}, S_{medium}])$ > input for $S_{fine}$ | Hinge loss (Margin Loss) **Without** <br />Consistency ($L_{cons}$) |
| Mod-3.3 | Top-Down <br   />[Coarse-to-Fine] | 8D | 64D>32D>16D <br   />(Coarse>Medium>FINE) <br />use skip connections between   Secondary Capsules <br />$Concatenate([P_{caps}, S_{coarse}])$ >   input for $S_{medium}$ <br />$Concatenate([P_{caps}, S_{medium}])$ >   input for $S_{fine}$ |   Hinge loss (Margin Loss)   **With** <br />Consistency ($L_{cons}$)  |
| Mod-3.4 | Top-Down <br   />[Coarse-to-Fine] | 8D | 64D>32D>16D <br   />(Coarse>Medium>FINE) <br />use skip connections between   Secondary Capsules <br />$Concatenate([P_{caps}, S_{coarse}])$ >   input for $S_{medium}$ <br />$Concatenate([P_{caps}, S_{medium}])$ >   input for $S_{fine}$ | Hinge loss (Margin Loss)   **Without** <br />Consistency ($L_{cons}$) |

**For training the model without $L_{cons}$ just applied Hinge loss (Margin Loss) for each level**

# Results:

***
|     Dataset     |  Models | Total  Trainable<br>     params (M) | Accuracy Coarse | Accuracy Medium | Accuracy Fine | Hierarchical Precision | Hierarchical Recall | Hierarchical F1-Score | Consistency | Exact Match |
|:---------------:|:-------:|:-----------------------------------:|:---------------:|:---------------:|:-------------:|:----------------------:|:-------------------:|:---------------------:|:-----------:|:-----------:|
|      EMNIST     | Mod-3.1 |                 4.91                |      94.32%     |                 |     89.28%    |         91.80%         |        92.01%       |         91.88%        |    99.15%   |    88.89%   |
|      EMNIST     | Mod-3.2 |                 4.91                |      94.22%     |                 |     89.39%    |         91.81%         |        91.98%       |         91.88%        |    99.30%   |    89.06%   |
|      EMNIST     | Mod-3.3 |                 5.15                |      94.27%     |                 |     89.32%    |         91.80%         |        92.00%       |         91.88%        |    99.21%   |    88.93%   |
|      EMNIST     | Mod-3.4 |                 5.15                |      94.44%     |                 |     89.88%    |         92.17%         |        92.32%       |         92.23%        |    99.40%   |    89.57%   |
|                 |         |                                     |                 |                 |               |                        |                     |                       |             |             |
| Fashion   MNIST | Mod-3.1 |                 4.82                |      99.92%     |      97.79%     |     94.83%    |         97.51%         |        97.54%       |         97.52%        |    99.84%   |    94.74%   |
| Fashion   MNIST | Mod-3.2 |                 4.82                |      99.89%     |      97.78%     |     94.92%    |         97.53%         |        97.59%       |         97.55%        |    99.70%   |    94.77%   |
| Fashion   MNIST | Mod-3.3 |                 4.99                |      99.90%     |      97.69%     |     94.64%    |         97.38%         |        97.44%       |         97.41%        |    99.69%   |    94.55%   |
| Fashion   MNIST | Mod-3.4 |                 4.99                |      99.92%     |      97.82%     |     94.95%    |         97.55%         |        97.59%       |         97.57%        |    99.78%   |    94.88%   |
|                 |         |                                     |                 |                 |               |                        |                     |                       |             |             |
|     CIFAR-10    | Mod-0.1 |                 5.04                |      98.81%     |      93.80%     |     90.99%    |         94.47%         |        94.63%       |         94.53%        |    99.08%   |    90.75%   |
|     CIFAR-10    | Mod-0.2 |                 5.04                |      98.72%     |      93.81%     |     90.84%    |         94.41%         |        94.59%       |         94.48%        |    99.06%   |    90.56%   |
|     CIFAR-10    | Mod-2.1 |                 4.84                |      98.76%     |      93.36%     |     90.26%    |         94.09%         |        94.30%       |         94.18%        |    98.94%   |    89.85%   |
|     CIFAR-10    | Mod-2.2 |                 4.84                |      97.88%     |      89.79%     |     85.83%    |         91.12%         |        91.41%       |         91.24%        |    98.44%   |    85.28%   |
|     CIFAR-10    | Mod-2.3 |                 4.84                |      98.33%     |      91.13%     |     87.65%    |         92.32%         |        92.54%       |         92.41%        |    98.75%   |    87.25%   |
|     CIFAR-10    | Mod-2.4 |                 4.84                |      98.24%     |      90.96%     |     87.24%    |         92.09%         |        92.35%       |         92.20%        |    98.62%   |    86.75%   |
|     CIFAR-10    | Mod-2.5 |                 4.84                |      98.41%     |      91.42%     |     87.96%    |         92.55%         |        92.80%       |         92.66%        |    98.65%   |    87.47%   |
|     CIFAR-10    | Mod-2.6 |                 4.84                |      98.67%     |      92.74%     |     89.45%    |         93.58%         |        93.74%       |         93.64%        |    99.10%   |    89.16%   |
|     CIFAR-10    | Mod-2.7 |                 4.86                |      98.68%     |      93.58%     |     90.77%    |         94.32%         |        94.44%       |         94.37%        |    99.30%   |    90.55%   |
|     CIFAR-10    | Mod-2.8 |                 4.86                |      98.40%     |      92.11%     |     88.67%    |         93.03%         |        93.26%       |         93.12%        |    98.84%   |    88.17%   |
|     CIFAR-10    | Mod-3.1 |                 5.23                |      98.79%     |      94.28%     |     91.22%    |         94.74%         |        94.89%       |         94.80%        |    99.18%   |    90.95%   |
|     CIFAR-10    | Mod-3.2 |                 5.23                |      98.71%     |      94.01%     |     90.97%    |         94.53%         |        94.73%       |         94.62%        |    98.99%   |    90.58%   |
|     CIFAR-10    | Mod-3.3 |                 5.80                |      98.82%     |      93.96%     |     91.26%    |         94.65%         |        94.82%       |         94.72%        |    99.11%   |    90.90%   |
|     CIFAR-10    | Mod-3.4 |                 5.80                |      98.73%     |      94.15%     |     91.24%    |         94.65%         |        94.83%       |         94.73%        |    98.98%   |    90.93%   |
|                 |         |                                     |                 |                 |               |                        |                     |                       |             |             |
|    CIFAR-100    | Mod-0.1 |                 8.52                |      86.85%     |      79.14%     |     66.18%    |         77.07%         |        78.82%       |         77.75%        |    90.37%   |    64.08%   |
|    CIFAR-100    | Mod-2.1 |                 5.55                |      86.57%     |      78.33%     |     57.08%    |         73.86%         |        75.00%       |         74.31%        |    92.51%   |    56.10%   |
|    CIFAR-100    | Mod-2.2 |                 5.55                |      87.40%     |      79.22%     |     58.31%    |         74.70%         |        75.98%       |         75.20%        |    91.76%   |    57.00%   |
|    CIFAR-100    | Mod-2.3 |                 5.55                |      86.03%     |      77.48%     |     57.01%    |         73.34%         |        74.73%       |         73.88%        |    90.90%   |    55.79%   |
|    CIFAR-100    | Mod-2.4 |                 5.55                |      86.78%     |      78.88%     |     56.86%    |         73.97%         |        75.40%       |         74.52%        |    91.03%   |    55.68%   |
|    CIFAR-100    | Mod-2.7 |                 6.10                |      86.15%     |      77.45%     |     61.18%    |         74.75%         |        76.25%       |         75.33%        |    90.68%   |    59.52%   |
|    CIFAR-100    | Mod-2.8 |                 6.10                |      85.36%     |      76.63%     |     60.07%    |         73.82%         |        75.38%       |         74.42%        |    89.86%   |    58.51%   |
|    CIFAR-100    | Mod-3.1 |                 7.85                |      86.93%     |      79.31%     |     66.38%    |         77.43%         |        79.20%       |         78.12%        |    89.80%   |    63.80%   |
|    CIFAR-100    | Mod-3.2 |                 7.85                |      86.81%     |      78.73%     |     66.23%    |         77.84%         |        79.56%       |         78.52%        |    89.78%   |    64.41%   |
|    CIFAR-100    | Mod-3.3 |                11.68                |      86.95%     |      78.76%     |     66.06%    |         77.07%         |        78.99%       |         77.82%        |    88.85%   |    63.33%   |
|    CIFAR-100    | Mod-3.4 |                11.68                |      87.27%     |      79.06%     |     66.61%    |         77.40%         |        79.30%       |         78.14%        |    88.85%   |    63.93%   |
|                 |         |                                     |                 |                 |               |                        |                     |                       |             |             |
|     CU Bird     | Mod-3.1 |                49.75                |      41.49%     |      21.95%     |     15.43%    |         24.73%         |        31.47%       |         27.27%        |    28.06%   |    8.94%    |
|     CU Bird     | Mod-3.2 |                49.75                |      41.53%     |      22.92%     |     15.22%    |         24.89%         |        31.54%       |         27.38%        |    28.34%   |    9.29%    |
|     CU Bird     | Mod-3.3 |                106.01               |      38.45%     |      20.71%     |     12.53%    |         22.47%         |        29.03%       |         24.91%        |    25.65%   |    7.66%    |
|     CU Bird     | Mod-3.4 |                106.01               |      48.15%     |      30.48%     |     21.16%    |         31.60%         |        38.56%       |         34.24%        |    34.76%   |    14.15%   |
|                 |         |                                     |                 |                 |               |                        |                     |                       |             |             |
|  Stanford Cars  | Mod-3.1 |                39.34                |      49.29%     |      18.70%     |     13.50%    |         25.54%         |        33.27%       |         28.46%        |    26.91%   |    7.44%    |
|  Stanford Cars  | Mod-3.2 |                39.34                |      48.84%     |      19.37%     |     15.24%    |         26.46%         |        34.59%       |         29.53%        |    25.81%   |    8.51%    |
|  Stanford Cars  | Mod-3.3 |                81.17                |      53.34%     |      19.52%     |     14.05%    |         26.73%         |        34.69%       |         29.73%        |    29.15%   |    8.13%    |
|  Stanford Cars  | Mod-3.4 |                81.17                |      47.50%     |      16.39%     |     11.74%    |         23.56%         |        31.40%       |         26.50%        |    25.76%   |    6.19%    |
			
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
	