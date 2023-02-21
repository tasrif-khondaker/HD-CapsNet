# Modification-2 of HD-CapsNet
Hierarchical Deep Capsule Network For Image Classification. The model employs capsule network for each hierarchical levels, treating as a global classification model.
 
**The original architecture of the HD-CapsNet is as follows:**
 
![fig:Architecture](Results/Architecture.png?raw=true "Architecture of the HD-CapsNet")
 
## The following changes are made in the architecture.

| Modifications |             Approach             | Dimention of primary capsule <br   />($P$) |                                                                                           Dimention of <br />Secondary capsule   <br />($S_{i}$)                                                                                          |                            Loss Function                            |
|:-------------:|:--------------------------------:|:------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------:|
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
| Mod-3.3 | Top-Down <br   />[Coarse-to-Fine] | 8D | 32D>24D>16D <br   />(Coarse>Medium>FINE) <br />use skip connections between   Secondary Capsules <br />$Concatenate([P_{caps}, S_{coarse}])$ >   input for $S_{medium}$ <br />$Concatenate([P_{caps}, S_{medium}])$ >   input for $S_{fine}$ |   Hinge loss (Margin Loss)   **With** <br />Consistency ($L_{cons}$)  |
|:-------:|:---------------------------------:|:--:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------:|
| Mod-3.4 | Top-Down <br   />[Coarse-to-Fine] | 8D | 32D>24D>16D <br   />(Coarse>Medium>FINE) <br />use skip connections between   Secondary Capsules <br />$Concatenate([P_{caps}, S_{coarse}])$ >   input for $S_{medium}$ <br />$Concatenate([P_{caps}, S_{medium}])$ >   input for $S_{fine}$ | Hinge loss (Margin Loss)   **Without** <br />Consistency ($L_{cons}$) |

**For training the model without $L_{cons}$ just applied Hinge loss (Margin Loss) for each level**

# Results:

***
<h3 align="center"> For CIFAR-10 Dataset </h3>

|                   Modifications                   | HD-CapsNet | HD-CapsNet <br />(W/O Lc) | Mod-1.1 | Mod-1.2 | Mod-1.3 | Mod-1.4 | Mod-2.1 | Mod-2.2 | Mod-2.3 | Mod-2.4 | Mod-2.5 | Mod-2.6 | Mod-2.7 | Mod-2.8 | Mod-3.1 | Mod-3.2 |
|:-------------------------------------------------:|:----------:|:-------------------------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| Total  Trainable <br />params (M)      params (M) |    5.04    |            5.04           |   5.03  |   4.74  |   5.03  |   4.74  |   4.84  |   4.84  |   4.84  |   4.84  |   4.84  |   4.84  |   4.86  |   4.86  |   5.23  |   5.23  |
|                  Accuracy Coarse                  |   98.81%   |           98.72%          |  98.71% |  98.24% |  98.81% |  98.62% |  98.76% |  97.88% |  98.33% |  98.24% |  98.41% |  98.67% |  98.68% |  98.40% |  98.79% |  98.71% |
|                  Accuracy Medium                  |   93.80%   |           93.81%          |  93.96% |  91.83% |  93.86% |  93.01% |  93.36% |  89.79% |  91.13% |  90.96% |  91.42% |  92.74% |  93.58% |  92.11% |  94.28% |  94.01% |
|                   Accuracy Fine                   |   90.99%   |           90.84%          |  91.16% |  87.89% |  91.03% |  89.67% |  90.26% |  85.83% |  87.65% |  87.24% |  87.96% |  89.45% |  90.77% |  88.67% |  91.22% |  90.97% |
|               Hierarchical Precision              |   94.47%   |           94.41%          |  94.56% |  92.57% |  94.55% |  93.73% |  94.09% |  91.12% |  92.32% |  92.09% |  92.55% |  93.58% |  94.32% |  93.03% |  94.74% |  94.53% |
|                Hierarchical Recall                |   94.63%   |           94.59%          |  94.72% |  92.86% |  94.66% |  93.94% |  94.30% |  91.41% |  92.54% |  92.35% |  92.80% |  93.74% |  94.44% |  93.26% |  94.89% |  94.73% |
|               Hierarchical F1-Score               |   94.53%   |           94.48%          |  94.62% |  92.69% |  94.59% |  93.82% |  94.18% |  91.24% |  92.41% |  92.20% |  92.66% |  93.64% |  94.37% |  93.12% |  94.80% |  94.62% |
|                    Consistency                    |   99.08%   |           99.06%          |  99.13% |  98.46% |  99.38% |  98.95% |  98.94% |  98.44% |  98.75% |  98.62% |  98.65% |  99.10% |  99.30% |  98.84% |  99.18% |  98.99% |
|                    Exact Match                    |   90.75%   |           90.56%          |  90.89% |  87.42% |  90.80% |  89.28% |  89.85% |  85.28% |  87.25% |  86.75% |  87.47% |  89.16% |  90.55% |  88.17% |  90.95% |  90.58% |

***
<h3 align="center"> For CIFAR-100 Dataset </h3>

|                   Modifications                   | HD-CapsNet | HD-CapsNet <br />(W/O Lc) | Mod-1.1 | Mod-1.2 | Mod-1.3 | Mod-1.4 | Mod-2.1 | Mod-2.2 | Mod-2.3 | Mod-2.4 | Mod-2.5 | Mod-2.6 | Mod-2.7 | Mod-2.8 | Mod-3.1 | Mod-3.2 |
|:-------------------------------------------------:|:----------:|:-------------------------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| Total  Trainable <br />params (M)      params (M) |    8.52    |            8.52           |   8.37  |   5.22  |   8.37  |   5.22  |   5.55  |   5.55  |   5.55  |   5.55  |         |         |   6.10  |   6.10  |   7.85  |   7.85  |
|                  Accuracy Coarse                  |   86.85%   |           86.03%          |  86.75% |  86.93% |  86.18% |  86.45% |  86.57% |  87.40% |  86.03% |  86.78% |         |         |  86.15% |  85.36% |  86.93% |  86.81% |
|                  Accuracy Medium                  |   79.14%   |           77.83%          |  78.95% |  78.73% |  78.31% |  77.78% |  78.33% |  79.22% |  77.48% |  78.88% |         |         |  77.45% |  76.63% |  79.31% |  78.73% |
|                   Accuracy Fine                   |   66.18%   |           64.87%          |  66.17% |  61.70% |  64.69% |  55.18% |  57.08% |  58.31% |  57.01% |  56.86% |         |         |  61.18% |  60.07% |  66.38% |  66.23% |
|               Hierarchical Precision              |   77.07%   |           76.04%          |  77.00% |  75.51% |  76.14% |  73.12% |  73.86% |  74.70% |  73.34% |  73.97% |         |         |  74.75% |  73.82% |  77.43% |  77.84% |
|                Hierarchical Recall                |   78.82%   |           77.87%          |  78.73% |  76.97% |  77.98% |  74.26% |  75.00% |  75.98% |  74.73% |  75.40% |         |         |  76.25% |  75.38% |  79.20% |  79.56% |
|               Hierarchical F1-Score               |   77.75%   |           76.75%          |  77.68% |  76.07% |  76.86% |  73.57% |  74.31% |  75.20% |  73.88% |  74.52% |         |         |  75.33% |  74.42% |  78.12% |  78.52% |
|                    Consistency                    |   90.37%   |           89.81%          |  90.00% |  90.64% |  89.83% |  92.64% |  92.51% |  91.76% |  90.90% |  91.03% |         |         |  90.68% |  89.86% |  89.80% |  89.78% |
|                    Exact Match                    |   64.08%   |           62.53%          |  64.04% |  60.44% |  62.39% |  53.93% |  56.10% |  57.00% |  55.79% |  55.68% |         |         |  59.52% |  58.51% |  63.80% |  64.41% |
			
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
	