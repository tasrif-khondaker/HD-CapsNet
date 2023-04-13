# Modification-2 of HD-CapsNet
Hierarchical Deep Capsule Network For Image Classification. The model employs capsule network for each hierarchical levels, treating as a global classification model.
 
<!-- **The architecture of the BUH-CapsNet with Consistency loss is as follows:**
 
![fig:Architecture](Results/Architecture_BUH_CapsNet.png?raw=true "Architecture of the BUH-CapsNet") -->

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
|      EMNIST     | Mod-3.1 |                 4.91                |      94.32%     |        --       |     89.28%    |         91.80%         |        92.01%       |         91.88%        |    99.15%   |    88.89%   |
|      EMNIST     | Mod-3.2 |                 4.91                |      94.22%     |        --       |     89.39%    |         91.81%         |        91.98%       |         91.88%        |    99.30%   |    89.06%   |
|                 |         |                                     |                 |                 |               |                        |                     |                       |             |             |
| Fashion   MNIST | Mod-3.1 |                 4.82                |      99.92%     |      97.79%     |     94.83%    |         97.51%         |        97.54%       |         97.52%        |    99.84%   |    94.74%   |
| Fashion   MNIST | Mod-3.2 |                 4.82                |      99.89%     |      97.78%     |     94.92%    |         97.53%         |        97.59%       |         97.55%        |    99.70%   |    94.77%   |
|                 |         |                                     |                 |                 |               |                        |                     |                       |             |             |
|     CIFAR-10    | Mod-3.1 |                 5.23                |      98.79%     |      94.28%     |     91.22%    |         94.74%         |        94.89%       |         94.80%        |    99.18%   |    90.95%   |
|     CIFAR-10    | Mod-3.2 |                 5.23                |      98.71%     |      94.01%     |     90.97%    |         94.53%         |        94.73%       |         94.62%        |    98.99%   |    90.58%   |
|                 |         |                                     |                 |                 |               |                        |                     |                       |             |             |
|    CIFAR-100    | Mod-3.1 |                 7.85                |      86.93%     |      79.31%     |     66.38%    |         77.43%         |        79.20%       |         78.12%        |    89.80%   |    63.80%   |
|    CIFAR-100    | Mod-3.2 |                 7.85                |      86.81%     |      78.73%     |     66.23%    |         77.84%         |        79.56%       |         78.52%        |    89.78%   |    64.41%   |
|                 |         |                                     |                 |                 |               |                        |                     |                       |             |             |
|     CU Bird     | Mod-3.3 |                106.01               |      40.42%     |      21.61%     |     13.39%    |         23.47%         |        30.33%       |         26.01%        |    27.34%   |    8.63%    |
|     CU Bird     | Mod-3.4 |                106.01               |      36.59%     |      17.78%     |     10.87%    |         20.29%         |        26.56%       |         22.62%        |    24.09%   |    6.28%    |
|                 |         |                                     |                 |                 |               |                        |                     |                       |             |             |
|  Stanford Cars  | Mod-3.3 |                81.17                |      53.34%     |      19.52%     |     14.05%    |         26.73%         |        34.69%       |         29.73%        |    29.15%   |    8.13%    |
|  Stanford Cars  | Mod-3.4 |                81.17                |      47.50%     |      16.39%     |     11.74%    |         23.56%         |        31.40%       |         26.50%        |    25.76%   |    6.19%    |
|	|	|	|	|	|	|	|	|	|	|	|
| Marine Tree | Mod-3.3 | 13.58 | 89.88% | 78.60% | 57.15% | 75.02% | 76.04% | 75.44% | 94.47% | 55.59% |
| Marine Tree | Mod-3.4 | 13.58 | 89.50% | 77.57% | 53.75% | 73.29% | 74.76% | 73.88% | 92.37% | 51.85% |

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
	