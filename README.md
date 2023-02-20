# Modification of HD-CapsNet
Hierarchical Deep Capsule Network For Image Classification. The model employs capsule network for each hierarchical levels, treating as a global classification model.
 
**The original architecture of the HD-CapsNet is as follows:**
 
![fig:Architecture](Results/Architecture.png?raw=true "Architecture of the HD-CapsNet")
 
## The following changes are made in the architecture.
- Changing the Secondary Capsule Dimension:
	1. 8D>12D>16D (Coarse>Medium>FINE)
		1. Applying top-down approach (Coarse-to-Fine)
		2. Applying bottom-up approach (Fine-to-Coarse)
		
### For both modifications, we train the model with and without the consistency loss $L_{cons}$ function

**For training the model without $L_{cons}$ just applied Hinge loss (Margin Loss) for each level**

# Results:
### Here the following Modification applies:

- Mod-1.1       = Bottom-Up (BU) Approach With $L_{cons}$    \[Fine-to-Coarse\]		
- Mod-1.2	= Top-Down (TD) approach With $L_{cons}$	 \[Coarse-to-Fine\]			
- Mod-1.3	= Bottom-Up (BU) Approach Without $L_{cons}$ \[Fine-to-Coarse\]				
- Mod-1.4	= Bottom-Up (TD) Approach Without $L_{cons}$ \[Coarse-to-Fine\]	

***
<h3 align="center"> For CIFAR-10 Dataset </h3>

| Parameters                        | HD-CapsNet | HD-CapsNet (W/O Lc) | Mod-1.1 | Mod-1.2 | Mod-1.3 | Mod-1.4 |
|-----------------------------------|------------|---------------------|---------|---------|---------|---------|
| Total  Trainable       params (M) | 5.04       | 5.04                | 5.03    | 4.74    | 5.03    | 4.74    |
| Accuracy Coarse                   | 98.81%     | 98.72%              | 98.71%  | 98.24%  | 98.81%  | 98.62%  |
| Accuracy Medium                   | 93.80%     | 93.81%              | 93.96%  | 91.83%  | 93.86%  | 93.01%  |
| Accuracy Fine                     | 90.99%     | 90.84%              | 91.16%  | 87.89%  | 91.03%  | 89.67%  |
| Hierarchical Precision            | 94.47%     | 94.41%              | 94.56%  | 92.57%  | 94.55%  | 93.73%  |
| Hierarchical Recall               | 94.63%     | 94.59%              | 94.72%  | 92.86%  | 94.66%  | 93.94%  |
| Hierarchical F1-Score             | 94.53%     | 94.48%              | 94.62%  | 92.69%  | 94.59%  | 93.82%  |
| Consistency                       | 99.08%     | 99.06%              | 99.13%  | 98.46%  | 99.38%  | 98.95%  |
| Exact Match                       | 90.75%     | 90.56%              | 90.89%  | 87.42%  | 90.80%  | 89.28%  |

***
<h3 align="center"> For CIFAR-100 Dataset </h3>

|            Parameters            | HD-CapsNet | HD-CapsNet (W/O Lc) | Mod-1.1 | Mod-1.2 | Mod-1.3 | Mod-1.4 |
|:--------------------------------:|:----------:|:-------------------:|:-------:|:-------:|:-------:|:-------:|
| Total  Trainable      params (M) |    8.52    |         8.52        |   8.37  |   5.22  |   8.37  |   5.22  |
|          Accuracy Coarse         |   86.85%   |        86.03%       |  86.75% |  86.93% |  86.18% |  86.45% |
|          Accuracy Medium         |   79.14%   |        77.83%       |  78.95% |  78.73% |  78.31% |  77.78% |
|           Accuracy Fine          |   66.18%   |        64.87%       |  66.17% |  61.70% |  64.69% |  55.18% |
|      Hierarchical Precision      |   77.07%   |        76.04%       |  77.00% |  75.51% |  76.14% |  73.12% |
|        Hierarchical Recall       |   78.82%   |        77.87%       |  78.73% |  76.97% |  77.98% |  74.26% |
|       Hierarchical F1-Score      |   77.75%   |        76.75%       |  77.68% |  76.07% |  76.86% |  73.57% |
|            Consistency           |   90.37%   |        89.81%       |  90.00% |  90.64% |  89.83% |  92.64% |
|            Exact Match           |   64.08%   |        62.53%       |  64.04% |  60.44% |  62.39% |  53.93% |

			
