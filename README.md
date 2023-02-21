# Modification-2 of HD-CapsNet
Hierarchical Deep Capsule Network For Image Classification. The model employs capsule network for each hierarchical levels, treating as a global classification model.
 
**The original architecture of the HD-CapsNet is as follows:**
 
![fig:Architecture](Results/Architecture.png?raw=true "Architecture of the HD-CapsNet")
 
## The following changes are made in the architecture.
| Modifications |              Approach              | Dimention of <br />primary   capsule <br />($P$) | Dimention of <br   />Secondary capsule <br />($S_{i}$) |                             Loss Function                             |
|:-------------:|:----------------------------------:|:------------------------------------------------:|:------------------------------------------------------:|:---------------------------------------------------------------------:|
| Modifications | Approach                         | Dimention of <br />primary capsule   <br />($P$) | Dimention of <br />Secondary capsule   <br />($S_{i}$) | Loss Function                                                       |
|---------------|----------------------------------|--------------------------------------------------|--------------------------------------------------------|---------------------------------------------------------------------|
| Mod-1.1       | Bottom-UP <br />[Fine-to-Coarse] | 8D                                               | 16D>12D>8D <br />(Fine>Medium>Coarse)                  | Hinge loss (Margin Loss) **With** <br />Consistency ($L_{cons}$)    |
| Mod-1.2       | Top-Down <br />[Coarse-to-Fine]  | 8D                                               | 8D>12D>16D <br />(Coarse>Medium>FINE)                  | Hinge loss (Margin Loss) **With** <br />Consistency ($L_{cons}$)    |
| Mod-1.3       | Bottom-UP <br />[Fine-to-Coarse] | 8D                                               | 16D>12D>8D <br />(Fine>Medium>Coarse)                  | Hinge loss (Margin Loss) **Without** <br />Consistency ($L_{cons}$) |
| Mod-1.4       | Top-Down <br />[Coarse-to-Fine]  | 8D                                               | 8D>12D>16D <br />(Coarse>Medium>FINE)                  | Hinge loss (Margin Loss) **Without** <br />Consistency ($L_{cons}$) |
| Mod-2.1       | Top-Down <br />[Coarse-to-Fine]  | 8D                                               | 32D>16D>8D <br />(Coarse>Medium>FINE)                  | Hinge loss (Margin Loss) **With** <br />Consistency ($L_{cons}$)    |
| Mod-2.2       | Top-Down <br />[Coarse-to-Fine]  | 16D                                              | 32D>16D>8D <br />(Coarse>Medium>FINE)                  | Hinge loss (Margin Loss) **With** <br />Consistency ($L_{cons}$)    |
| Mod-2.3       | Top-Down <br />[Coarse-to-Fine]  | 8D                                               | 32D>16D>8D <br />(Coarse>Medium>FINE)                  | Hinge loss (Margin Loss) **Without** <br />Consistency ($L_{cons}$) |
| Mod-2.4       | Top-Down <br />[Coarse-to-Fine]  | 16D                                              | 32D>16D>8D <br />(Coarse>Medium>FINE)                  | Hinge loss (Margin Loss) **Without** <br />Consistency ($L_{cons}$) |
| Mod-2.5       | Top-Down <br />[Coarse-to-Fine]  | 4D                                               | 32D>16D>8D <br />(Coarse>Medium>FINE)                  | Hinge loss (Margin Loss) **With** <br />Consistency ($L_{cons}$)    |
| Mod-2.6       | Top-Down <br />[Coarse-to-Fine]  | 4D                                               | 32D>16D>8D <br />(Coarse>Medium>FINE)                  | Hinge loss (Margin Loss) **Without** <br />Consistency ($L_{cons}$) |

**For training the model without $L_{cons}$ just applied Hinge loss (Margin Loss) for each level**

# Results:

***
<h3 align="center"> For CIFAR-10 Dataset </h3>

|            Parameters            | HD-CapsNet | HD-CapsNet <br />(W/O Lc) | Mod-1.1 | Mod-1.2 | Mod-1.3 | Mod-1.4 |
|:--------------------------------:|:----------:|:-------------------:|:-------:|:-------:|:-------:|:-------:|
| Total  Trainable <br />params (M) |    5.04    |         5.04        |   5.03  |   4.74  |   5.03  |   4.74  |
|          Accuracy Coarse         |   98.81%   |        98.72%       |  98.71% |  98.24% |  98.81% |  98.62% |
|          Accuracy Medium         |   93.80%   |        93.81%       |  93.96% |  91.83% |  93.86% |  93.01% |
|           Accuracy Fine          |   90.99%   |        90.84%       |  91.16% |  87.89% |  91.03% |  89.67% |
|      Hierarchical Precision      |   94.47%   |        94.41%       |  94.56% |  92.57% |  94.55% |  93.73% |
|        Hierarchical Recall       |   94.63%   |        94.59%       |  94.72% |  92.86% |  94.66% |  93.94% |
|       Hierarchical F1-Score      |   94.53%   |        94.48%       |  94.62% |  92.69% |  94.59% |  93.82% |
|            Consistency           |   99.08%   |        99.06%       |  99.13% |  98.46% |  99.38% |  98.95% |
|            Exact Match           |   90.75%   |        90.56%       |  90.89% |  87.42% |  90.80% |  89.28% |

***
<h3 align="center"> For CIFAR-100 Dataset </h3>

|            Parameters            | HD-CapsNet | HD-CapsNet <br />(W/O Lc) | Mod-1.1 | Mod-1.2 | Mod-1.3 | Mod-1.4 |
|:--------------------------------:|:----------:|:-------------------:|:-------:|:-------:|:-------:|:-------:|
| Total  Trainable <br />params (M) |    8.52    |         8.52        |   8.37  |   5.22  |   8.37  |   5.22  |
|          Accuracy Coarse         |   86.85%   |        86.03%       |  86.75% |  86.93% |  86.18% |  86.45% |
|          Accuracy Medium         |   79.14%   |        77.83%       |  78.95% |  78.73% |  78.31% |  77.78% |
|           Accuracy Fine          |   66.18%   |        64.87%       |  66.17% |  61.70% |  64.69% |  55.18% |
|      Hierarchical Precision      |   77.07%   |        76.04%       |  77.00% |  75.51% |  76.14% |  73.12% |
|        Hierarchical Recall       |   78.82%   |        77.87%       |  78.73% |  76.97% |  77.98% |  74.26% |
|       Hierarchical F1-Score      |   77.75%   |        76.75%       |  77.68% |  76.07% |  76.86% |  73.57% |
|            Consistency           |   90.37%   |        89.81%       |  90.00% |  90.64% |  89.83% |  92.64% |
|            Exact Match           |   64.08%   |        62.53%       |  64.04% |  60.44% |  62.39% |  53.93% |

			
