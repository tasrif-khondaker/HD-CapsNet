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
|    Mod-2.1    |  Top-Down <br />[Coarse-to-Fine] |                     8D                     | 32D>16D>8D <br />(Coarse>Medium>FINE) <br />[WITHOUT SKIP CONNECTION]                                                                                                |   Hinge loss (Margin Loss) **With** <br />Consistency ($L_{cons}$)  |
|    Mod-2.2    |  Top-Down <br />[Coarse-to-Fine] |                     16D                    |                                                                                                   32D>16D>8D <br />(Coarse>Medium>FINE)                                                                                                   |   Hinge loss (Margin Loss) **With** <br />Consistency ($L_{cons}$)  |
|    Mod-2.3    |  Top-Down <br />[Coarse-to-Fine] |                     8D                     |                                                                                                   32D>16D>8D <br />(Coarse>Medium>FINE)                                                                                                   | Hinge loss (Margin Loss) **Without** <br />Consistency ($L_{cons}$) |
|    Mod-2.4    |  Top-Down <br />[Coarse-to-Fine] |                     16D                    |                                                                                                   32D>16D>8D <br />(Coarse>Medium>FINE)                                                                                                   | Hinge loss (Margin Loss) **Without** <br />Consistency ($L_{cons}$) |
|    Mod-2.5    |  Top-Down <br />[Coarse-to-Fine] |                     4D                     |                                                                                                   32D>16D>8D <br />(Coarse>Medium>FINE)                                                                                                   |   Hinge loss (Margin Loss) **With** <br />Consistency ($L_{cons}$)  |
|    Mod-2.6    |  Top-Down <br />[Coarse-to-Fine] |                     4D                     |                                                                                                   32D>16D>8D <br />(Coarse>Medium>FINE)                                                                                                   | Hinge loss (Margin Loss) **Without** <br />Consistency ($L_{cons}$) |
|    Mod-2.7    |  Top-Down <br />[Coarse-to-Fine] |                     8D                     |                                                                                                   32D>24D>16D <br />(Coarse>Medium>FINE)                                                                                                  |   Hinge loss (Margin Loss) **With** <br />Consistency ($L_{cons}$)  |
|    Mod-2.8    |  Top-Down <br />[Coarse-to-Fine] |                     8D                     |                                                                                                   32D>24D>16D <br />(Coarse>Medium>FINE)                                                                                                  | Hinge loss (Margin Loss) **Without** <br />Consistency ($L_{cons}$) |
| Mod-2.9 | Top-Down <br />[Coarse-to-Fine] | 8D | 64D>32D>16D <br />(Coarse>Medium>FINE)<br />[WITHOUT SKIP CONNECTION] | Hinge loss (Margin Loss) **With** <br />Consistency ($L_{cons}$) |
|    Mod-3.1    |  Top-Down <br />[Coarse-to-Fine] |                     8D                     | 32D>16D>8D <br />(Coarse>Medium>FINE) <br />use   skip connections between Secondary Capsules <br   />$Concatenate([P_{caps}, S_{coarse}])$ > input for $S_{medium}$ <br   />$Concatenate([P_{caps}, S_{medium}])$ > input for $S_{fine}$ |   Hinge loss (Margin Loss) **With** <br />Consistency ($L_{cons}$)  |
|    Mod-3.2    |  Top-Down <br />[Coarse-to-Fine] |                     8D                     | 32D>16D>8D <br />(Coarse>Medium>FINE) <br />use   skip connections between Secondary Capsules <br   />$Concatenate([P_{caps}, S_{coarse}])$ > input for $S_{medium}$ <br   />$Concatenate([P_{caps}, S_{medium}])$ > input for $S_{fine}$ | Hinge loss (Margin Loss) **Without** <br />Consistency ($L_{cons}$) |
| Mod-3.3 | Top-Down <br   />[Coarse-to-Fine] | 8D | 64D>32D>16D <br   />(Coarse>Medium>FINE) <br />use skip connections between   Secondary Capsules <br />$Concatenate([P_{caps}, S_{coarse}])$ >   input for $S_{medium}$ <br />$Concatenate([P_{caps}, S_{medium}])$ >   input for $S_{fine}$ |   Hinge loss (Margin Loss)   **With** <br />Consistency ($L_{cons}$)  |
| Mod-3.4 | Top-Down <br   />[Coarse-to-Fine] | 8D | 64D>32D>16D <br   />(Coarse>Medium>FINE) <br />use skip connections between   Secondary Capsules <br />$Concatenate([P_{caps}, S_{coarse}])$ >   input for $S_{medium}$ <br />$Concatenate([P_{caps}, S_{medium}])$ >   input for $S_{fine}$ | Hinge loss (Margin Loss)   **Without** <br />Consistency ($L_{cons}$) |


**For training the model without $L_{cons}$ just applied Hinge loss (Margin Loss) for each level**

# Results:

***
| Dataset       | Models     | Model Description                        | Total  Trainable<br/>params (M) | Accuracy Coarse | Accuracy Medium | Accuracy Fine | Hierarchical Precision | Hierarchical Recall | Hierarchical F1-Score | Consistency | Exact Match |
|---------------|------------|------------------------------------------|---------------------------------|-----------------|-----------------|---------------|------------------------|---------------------|-----------------------|-------------|-------------|
| EMNIST        | Mod-3.1    | HD-CapsNet With Consistency Loss (Lc)    | 4.91                            | 94.32%          | --              | 89.28%        | 91.80%                 | 92.01%              | 91.88%                | 99.15%      | 88.89%      |
| EMNIST        | Mod-3.2    | HD-CapsNet Without Consistency Loss (Lc) | 4.91                            | 94.22%          | --              | 89.39%        | 91.81%                 | 91.98%              | 91.88%                | 99.30%      | 89.06%      |
|               |            |                                          |                                 |                 |                 |               |                        |                     |                       |             |             |
| Fashion MNIST | Mod-2.1    | HD-CapsNet Without Skip Connection       | 4.73                            | 99.91%          | 97.63%          | 94.66%        | 97.40%                 | 97.42%              | 97.41%                | 99.87%      | 94.60%      |
| Fashion MNIST | Mod-3.1    | HD-CapsNet With Consistency Loss (Lc)    | 4.82                            | 99.92%          | 97.79%          | 94.83%        | 97.51%                 | 97.54%              | 97.52%                | 99.84%      | 94.74%      |
| Fashion MNIST | Mod-3.2    | HD-CapsNet Without Consistency Loss (Lc) | 4.82                            | 99.89%          | 97.78%          | 94.92%        | 97.53%                 | 97.59%              | 97.55%                | 99.70%      | 94.77%      |
|               |            |                                          |                                 |                 |                 |               |                        |                     |                       |             |             |
| CIFAR-10      | Mod-2.1    | HD-CapsNet Without Skip Connection       | 4.84                            | 98.76%          | 93.36%          | 90.26%        | 94.09%                 | 94.30%              | 94.18%                | 98.94%      | 89.85%      |
| CIFAR-10      | Mod-3.1    | HD-CapsNet With Consistency Loss (Lc)    | 5.23                            | 98.79%          | 94.28%          | 91.22%        | 94.74%                 | 94.89%              | 94.80%                | 99.18%      | 90.95%      |
| CIFAR-10      | Mod-3.2    | HD-CapsNet Without Consistency Loss (Lc) | 5.23                            | 98.71%          | 94.01%          | 90.97%        | 94.53%                 | 94.73%              | 94.62%                | 98.99%      | 90.58%      |
|               |            |                                          |                                 |                 |                 |               |                        |                     |                       |             |             |
| CIFAR-100     | Mod-2.1    | HD-CapsNet Without Skip Connection       | 5.55                            | 86.57%          | 78.33%          | 57.08%        | 73.86%                 | 75.00%              | 74.31%                | 92.51%      | 56.10%      |
| CIFAR-100     | Mod-3.1    | HD-CapsNet With Consistency Loss (Lc)    | 7.85                            | 86.93%          | 79.31%          | 66.38%        | 77.43%                 | 79.20%              | 78.12%                | 89.80%      | 63.80%      |
| CIFAR-100     | Mod-3.2    | HD-CapsNet Without Consistency Loss (Lc) | 7.85                            | 86.81%          | 78.73%          | 66.23%        | 77.10%                 | 79.02%              | 77.85%                | 88.62%      | 63.36%      |
| CIFAR-100     | Mod-3.2 T2 | HD-CapsNet Without Consistency Loss (Lc) | 7.85                            | 86.81%          | 78.73%          | 66.23%        | 77.84%                 | 79.56%              | 78.52%                | 89.78%      | 64.41%      |
|               |            |                                          |                                 |                 |                 |               |                        |                     |                       |             |             |
| CU Bird       | Mod-2.9    | HD-CapsNet Without Skip Connection       | 47.56                           | 35.66%          | 16.98%          | 2.14%         | 14.97%                 | 20.86%              | 17.13%                | 21.44%      | 1.55%       |
| CU Bird       | Mod-3.3    | HD-CapsNet With Consistency Loss (Lc)    | 106.01                          | 40.42%          | 21.61%          | 13.39%        | 23.47%                 | 30.33%              | 26.01%                | 27.34%      | 8.63%       |
| CU Bird       | Mod-3.4    | HD-CapsNet Without Consistency Loss (Lc) | 106.01                          | 36.59%          | 17.78%          | 10.87%        | 20.29%                 | 26.56%              | 22.62%                | 24.09%      | 6.28%       |
|               |            |                                          |                                 |                 |                 |               |                        |                     |                       |             |             |
| Stanford Cars | Mod-2.9    | HD-CapsNet Without Skip Connection       | 25.85                           | 46.01%          | 12.29%          | 1.57%         | 17.10%                 | 24.04%              | 19.79%                | 13.60%      | 0.87%       |
| Stanford Cars | Mod-3.3    | HD-CapsNet With Consistency Loss (Lc)    | 81.17                           | 53.34%          | 19.52%          | 14.05%        | 26.73%                 | 34.69%              | 29.73%                | 29.15%      | 8.13%       |
| Stanford Cars | Mod-3.4    | HD-CapsNet Without Consistency Loss (Lc) | 81.17                           | 47.50%          | 16.39%          | 11.74%        | 23.56%                 | 31.40%              | 26.50%                | 25.76%      | 6.19%       |
|               |            |                                          |                                 |                 |                 |               |                        |                     |                       |             |             |
| Marine Tree   | Mod-2.9    | HD-CapsNet Without Skip Connection       | 5.97                            | 86.98%          | 77.82%          | 55.04%        | 73.35%                 | 75.76%              | 74.36%                | 86.95%      | 49.34%      |
| Marine Tree   | Mod-3.3    | HD-CapsNet With Consistency Loss (Lc)    | 13.58                           | 89.88%          | 78.60%          | 57.15%        | 75.02%                 | 76.04%              | 75.44%                | 94.47%      | 55.59%      |
| Marine Tree   | Mod-3.4    | HD-CapsNet Without Consistency Loss (Lc) | 13.58                           | 89.50%          | 77.57%          | 53.75%        | 73.29%                 | 74.76%              | 73.88%                | 92.37%      | 51.85%      |

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

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-pm97{background-color:#FCFCFF;border-color:inherit;text-align:left;vertical-align:top}
.tg .tg-7btt{border-color:inherit;font-weight:bold;text-align:center;vertical-align:top}
.tg .tg-0pky{border-color:inherit;text-align:left;vertical-align:top}
.tg .tg-x1e6{background-color:#F8696B;border-color:inherit;text-align:left;vertical-align:top}
.tg .tg-sfj8{background-color:#63BE7B;border-color:inherit;text-align:left;vertical-align:top}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-7btt">Dataset</th>
    <th class="tg-7btt">Models</th>
    <th class="tg-7btt">Model Description</th>
    <th class="tg-7btt">Total  Trainable<br>     params (M)</th>
    <th class="tg-7btt">Accuracy Coarse</th>
    <th class="tg-7btt">Accuracy Medium</th>
    <th class="tg-7btt">Accuracy Fine</th>
    <th class="tg-7btt">Hierarchical Precision</th>
    <th class="tg-7btt">Hierarchical Recall</th>
    <th class="tg-7btt">Hierarchical F1-Score</th>
    <th class="tg-7btt">Consistency</th>
    <th class="tg-7btt">Exact Match</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky">EMNIST</td>
    <td class="tg-0pky">Mod-3.1</td>
    <td class="tg-0pky">HD-CapsNet With Consistency Loss   (Lc)</td>
    <td class="tg-0pky">4.91</td>
    <td class="tg-0pky">94.32%</td>
    <td class="tg-0pky">--</td>
    <td class="tg-0pky">89.28%</td>
    <td class="tg-0pky">91.80%</td>
    <td class="tg-0pky">92.01%</td>
    <td class="tg-0pky">91.88%</td>
    <td class="tg-0pky">99.15%</td>
    <td class="tg-0pky">88.89%</td>
  </tr>
  <tr>
    <td class="tg-0pky">EMNIST</td>
    <td class="tg-0pky">Mod-3.2</td>
    <td class="tg-0pky">HD-CapsNet Without Consistency   Loss (Lc)</td>
    <td class="tg-0pky">4.91</td>
    <td class="tg-0pky">94.22%</td>
    <td class="tg-0pky">--</td>
    <td class="tg-0pky">89.39%</td>
    <td class="tg-0pky">91.81%</td>
    <td class="tg-0pky">91.98%</td>
    <td class="tg-0pky">91.88%</td>
    <td class="tg-0pky">99.30%</td>
    <td class="tg-0pky">89.06%</td>
  </tr>
  <tr>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">Fashion   MNIST</td>
    <td class="tg-0pky">Mod-2.1</td>
    <td class="tg-0pky">HD-CapsNet Without Skip   Connection</td>
    <td class="tg-0pky">4.73</td>
    <td class="tg-pm97"><span style="background-color:#FCFCFF">99.91%</span></td>
    <td class="tg-x1e6"><span style="background-color:#F8696B">97.63%</span></td>
    <td class="tg-x1e6"><span style="background-color:#F8696B">94.66%</span></td>
    <td class="tg-x1e6"><span style="background-color:#F8696B">97.40%</span></td>
    <td class="tg-x1e6"><span style="background-color:#F8696B">97.42%</span></td>
    <td class="tg-x1e6"><span style="background-color:#F8696B">97.41%</span></td>
    <td class="tg-sfj8"><span style="background-color:#63BE7B">99.87%</span></td>
    <td class="tg-x1e6"><span style="background-color:#F8696B">94.60%</span></td>
  </tr>
  <tr>
    <td class="tg-0pky">Fashion   MNIST</td>
    <td class="tg-0pky">Mod-3.1</td>
    <td class="tg-0pky">HD-CapsNet With Consistency Loss   (Lc)</td>
    <td class="tg-0pky">4.82</td>
    <td class="tg-sfj8"><span style="background-color:#63BE7B">99.92%</span></td>
    <td class="tg-sfj8"><span style="background-color:#63BE7B">97.79%</span></td>
    <td class="tg-pm97"><span style="background-color:#FCFCFF">94.83%</span></td>
    <td class="tg-pm97"><span style="background-color:#FCFCFF">97.51%</span></td>
    <td class="tg-pm97"><span style="background-color:#FCFCFF">97.54%</span></td>
    <td class="tg-pm97"><span style="background-color:#FCFCFF">97.52%</span></td>
    <td class="tg-pm97"><span style="background-color:#FCFCFF">99.84%</span></td>
    <td class="tg-pm97"><span style="background-color:#FCFCFF">94.74%</span></td>
  </tr>
  <tr>
    <td class="tg-0pky">Fashion   MNIST</td>
    <td class="tg-0pky">Mod-3.2</td>
    <td class="tg-0pky">HD-CapsNet Without Consistency   Loss (Lc)</td>
    <td class="tg-0pky">4.82</td>
    <td class="tg-x1e6"><span style="background-color:#F8696B">99.89%</span></td>
    <td class="tg-pm97"><span style="background-color:#FCFCFF">97.78%</span></td>
    <td class="tg-sfj8"><span style="background-color:#63BE7B">94.92%</span></td>
    <td class="tg-sfj8"><span style="background-color:#63BE7B">97.53%</span></td>
    <td class="tg-sfj8"><span style="background-color:#63BE7B">97.59%</span></td>
    <td class="tg-sfj8"><span style="background-color:#63BE7B">97.55%</span></td>
    <td class="tg-x1e6"><span style="background-color:#F8696B">99.70%</span></td>
    <td class="tg-sfj8"><span style="background-color:#63BE7B">94.77%</span></td>
  </tr>
  <tr>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">CIFAR-10</td>
    <td class="tg-0pky">Mod-2.1</td>
    <td class="tg-0pky">HD-CapsNet Without Skip   Connection</td>
    <td class="tg-0pky">4.84</td>
    <td class="tg-pm97"><span style="background-color:#FCFCFF">98.76%</span></td>
    <td class="tg-x1e6"><span style="background-color:#F8696B">93.36%</span></td>
    <td class="tg-x1e6"><span style="background-color:#F8696B">90.26%</span></td>
    <td class="tg-x1e6"><span style="background-color:#F8696B">94.09%</span></td>
    <td class="tg-x1e6"><span style="background-color:#F8696B">94.30%</span></td>
    <td class="tg-x1e6"><span style="background-color:#F8696B">94.18%</span></td>
    <td class="tg-x1e6"><span style="background-color:#F8696B">98.94%</span></td>
    <td class="tg-x1e6"><span style="background-color:#F8696B">89.85%</span></td>
  </tr>
  <tr>
    <td class="tg-0pky">CIFAR-10</td>
    <td class="tg-0pky">Mod-3.1</td>
    <td class="tg-0pky">HD-CapsNet With Consistency Loss   (Lc)</td>
    <td class="tg-0pky">5.23</td>
    <td class="tg-sfj8"><span style="background-color:#63BE7B">98.79%</span></td>
    <td class="tg-sfj8"><span style="background-color:#63BE7B">94.28%</span></td>
    <td class="tg-sfj8"><span style="background-color:#63BE7B">91.22%</span></td>
    <td class="tg-sfj8"><span style="background-color:#63BE7B">94.74%</span></td>
    <td class="tg-sfj8"><span style="background-color:#63BE7B">94.89%</span></td>
    <td class="tg-sfj8"><span style="background-color:#63BE7B">94.80%</span></td>
    <td class="tg-sfj8"><span style="background-color:#63BE7B">99.18%</span></td>
    <td class="tg-sfj8"><span style="background-color:#63BE7B">90.95%</span></td>
  </tr>
  <tr>
    <td class="tg-0pky">CIFAR-10</td>
    <td class="tg-0pky">Mod-3.2</td>
    <td class="tg-0pky">HD-CapsNet Without Consistency   Loss (Lc)</td>
    <td class="tg-0pky">5.23</td>
    <td class="tg-x1e6"><span style="background-color:#F8696B">98.71%</span></td>
    <td class="tg-pm97"><span style="background-color:#FCFCFF">94.01%</span></td>
    <td class="tg-pm97"><span style="background-color:#FCFCFF">90.97%</span></td>
    <td class="tg-pm97"><span style="background-color:#FCFCFF">94.53%</span></td>
    <td class="tg-pm97"><span style="background-color:#FCFCFF">94.73%</span></td>
    <td class="tg-pm97"><span style="background-color:#FCFCFF">94.62%</span></td>
    <td class="tg-pm97"><span style="background-color:#FCFCFF">98.99%</span></td>
    <td class="tg-pm97"><span style="background-color:#FCFCFF">90.58%</span></td>
  </tr>
  <tr>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">CIFAR-100</td>
    <td class="tg-0pky">Mod-2.1</td>
    <td class="tg-0pky">HD-CapsNet Without Skip   Connection</td>
    <td class="tg-0pky">5.55</td>
    <td class="tg-x1e6"><span style="background-color:#F8696B">86.57%</span></td>
    <td class="tg-x1e6"><span style="background-color:#F8696B">78.33%</span></td>
    <td class="tg-x1e6"><span style="background-color:#F8696B">57.08%</span></td>
    <td class="tg-x1e6"><span style="background-color:#F8696B">73.86%</span></td>
    <td class="tg-x1e6"><span style="background-color:#F8696B">75.00%</span></td>
    <td class="tg-x1e6"><span style="background-color:#F8696B">74.31%</span></td>
    <td class="tg-sfj8"><span style="background-color:#63BE7B">92.51%</span></td>
    <td class="tg-x1e6"><span style="background-color:#F8696B">56.10%</span></td>
  </tr>
  <tr>
    <td class="tg-0pky">CIFAR-100</td>
    <td class="tg-0pky">Mod-3.1</td>
    <td class="tg-0pky">HD-CapsNet With Consistency Loss   (Lc)</td>
    <td class="tg-0pky">7.85</td>
    <td class="tg-sfj8"><span style="background-color:#63BE7B">86.93%</span></td>
    <td class="tg-sfj8"><span style="background-color:#63BE7B">79.31%</span></td>
    <td class="tg-sfj8"><span style="background-color:#63BE7B">66.38%</span></td>
    <td class="tg-sfj8"><span style="background-color:#63BE7B">77.43%</span></td>
    <td class="tg-sfj8"><span style="background-color:#63BE7B">79.20%</span></td>
    <td class="tg-sfj8"><span style="background-color:#63BE7B">78.12%</span></td>
    <td class="tg-pm97"><span style="background-color:#FCFCFF">89.80%</span></td>
    <td class="tg-sfj8"><span style="background-color:#63BE7B">63.80%</span></td>
  </tr>
  <tr>
    <td class="tg-0pky">CIFAR-100</td>
    <td class="tg-0pky">Mod-3.2</td>
    <td class="tg-0pky">HD-CapsNet Without Consistency   Loss (Lc)</td>
    <td class="tg-0pky">7.85</td>
    <td class="tg-pm97"><span style="background-color:#FCFCFF">86.81%</span></td>
    <td class="tg-pm97"><span style="background-color:#FCFCFF">78.73%</span></td>
    <td class="tg-pm97"><span style="background-color:#FCFCFF">66.23%</span></td>
    <td class="tg-pm97"><span style="background-color:#FCFCFF">77.10%</span></td>
    <td class="tg-pm97"><span style="background-color:#FCFCFF">79.02%</span></td>
    <td class="tg-pm97"><span style="background-color:#FCFCFF">77.85%</span></td>
    <td class="tg-x1e6"><span style="background-color:#F8696B">88.62%</span></td>
    <td class="tg-pm97"><span style="background-color:#FCFCFF">63.36%</span></td>
  </tr>
  <tr>
    <td class="tg-0pky">CIFAR-100</td>
    <td class="tg-0pky">Mod-3.2 T2</td>
    <td class="tg-0pky">HD-CapsNet Without Consistency   Loss (Lc)</td>
    <td class="tg-0pky">7.85</td>
    <td class="tg-0pky">86.81%</td>
    <td class="tg-0pky">78.73%</td>
    <td class="tg-0pky">66.23%</td>
    <td class="tg-0pky">77.84%</td>
    <td class="tg-0pky">79.56%</td>
    <td class="tg-0pky">78.52%</td>
    <td class="tg-0pky">89.78%</td>
    <td class="tg-0pky">64.41%</td>
  </tr>
  <tr>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">CU Bird</td>
    <td class="tg-0pky">Mod-2.9</td>
    <td class="tg-0pky">HD-CapsNet Without Skip   Connection</td>
    <td class="tg-0pky">47.56</td>
    <td class="tg-x1e6"><span style="background-color:#F8696B">35.66%</span></td>
    <td class="tg-x1e6"><span style="background-color:#F8696B">16.98%</span></td>
    <td class="tg-x1e6"><span style="background-color:#F8696B">2.14%</span></td>
    <td class="tg-x1e6"><span style="background-color:#F8696B">14.97%</span></td>
    <td class="tg-x1e6"><span style="background-color:#F8696B">20.86%</span></td>
    <td class="tg-x1e6"><span style="background-color:#F8696B">17.13%</span></td>
    <td class="tg-x1e6"><span style="background-color:#F8696B">21.44%</span></td>
    <td class="tg-x1e6"><span style="background-color:#F8696B">1.55%</span></td>
  </tr>
  <tr>
    <td class="tg-0pky">CU Bird</td>
    <td class="tg-0pky">Mod-3.3</td>
    <td class="tg-0pky">HD-CapsNet With Consistency Loss   (Lc)</td>
    <td class="tg-0pky">106.01</td>
    <td class="tg-sfj8"><span style="background-color:#63BE7B">40.42%</span></td>
    <td class="tg-sfj8"><span style="background-color:#63BE7B">21.61%</span></td>
    <td class="tg-sfj8"><span style="background-color:#63BE7B">13.39%</span></td>
    <td class="tg-sfj8"><span style="background-color:#63BE7B">23.47%</span></td>
    <td class="tg-sfj8"><span style="background-color:#63BE7B">30.33%</span></td>
    <td class="tg-sfj8"><span style="background-color:#63BE7B">26.01%</span></td>
    <td class="tg-sfj8"><span style="background-color:#63BE7B">27.34%</span></td>
    <td class="tg-sfj8"><span style="background-color:#63BE7B">8.63%</span></td>
  </tr>
  <tr>
    <td class="tg-0pky">CU Bird</td>
    <td class="tg-0pky">Mod-3.4</td>
    <td class="tg-0pky">HD-CapsNet Without Consistency   Loss (Lc)</td>
    <td class="tg-0pky">106.01</td>
    <td class="tg-pm97"><span style="background-color:#FCFCFF">36.59%</span></td>
    <td class="tg-pm97"><span style="background-color:#FCFCFF">17.78%</span></td>
    <td class="tg-pm97"><span style="background-color:#FCFCFF">10.87%</span></td>
    <td class="tg-pm97"><span style="background-color:#FCFCFF">20.29%</span></td>
    <td class="tg-pm97"><span style="background-color:#FCFCFF">26.56%</span></td>
    <td class="tg-pm97"><span style="background-color:#FCFCFF">22.62%</span></td>
    <td class="tg-pm97"><span style="background-color:#FCFCFF">24.09%</span></td>
    <td class="tg-pm97"><span style="background-color:#FCFCFF">6.28%</span></td>
  </tr>
  <tr>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">Stanford Cars</td>
    <td class="tg-0pky">Mod-2.9</td>
    <td class="tg-0pky">HD-CapsNet Without Skip   Connection</td>
    <td class="tg-0pky">25.85</td>
    <td class="tg-x1e6"><span style="background-color:#F8696B">46.01%</span></td>
    <td class="tg-x1e6"><span style="background-color:#F8696B">12.29%</span></td>
    <td class="tg-x1e6"><span style="background-color:#F8696B">1.57%</span></td>
    <td class="tg-x1e6"><span style="background-color:#F8696B">17.10%</span></td>
    <td class="tg-x1e6"><span style="background-color:#F8696B">24.04%</span></td>
    <td class="tg-x1e6"><span style="background-color:#F8696B">19.79%</span></td>
    <td class="tg-x1e6"><span style="background-color:#F8696B">13.60%</span></td>
    <td class="tg-x1e6"><span style="background-color:#F8696B">0.87%</span></td>
  </tr>
  <tr>
    <td class="tg-0pky">Stanford Cars</td>
    <td class="tg-0pky">Mod-3.3</td>
    <td class="tg-0pky">HD-CapsNet With Consistency Loss   (Lc)</td>
    <td class="tg-0pky">81.17</td>
    <td class="tg-sfj8"><span style="background-color:#63BE7B">53.34%</span></td>
    <td class="tg-sfj8"><span style="background-color:#63BE7B">19.52%</span></td>
    <td class="tg-sfj8"><span style="background-color:#63BE7B">14.05%</span></td>
    <td class="tg-sfj8"><span style="background-color:#63BE7B">26.73%</span></td>
    <td class="tg-sfj8"><span style="background-color:#63BE7B">34.69%</span></td>
    <td class="tg-sfj8"><span style="background-color:#63BE7B">29.73%</span></td>
    <td class="tg-sfj8"><span style="background-color:#63BE7B">29.15%</span></td>
    <td class="tg-sfj8"><span style="background-color:#63BE7B">8.13%</span></td>
  </tr>
  <tr>
    <td class="tg-0pky">Stanford Cars</td>
    <td class="tg-0pky">Mod-3.4</td>
    <td class="tg-0pky">HD-CapsNet Without Consistency   Loss (Lc)</td>
    <td class="tg-0pky">81.17</td>
    <td class="tg-pm97"><span style="background-color:#FCFCFF">47.50%</span></td>
    <td class="tg-pm97"><span style="background-color:#FCFCFF">16.39%</span></td>
    <td class="tg-pm97"><span style="background-color:#FCFCFF">11.74%</span></td>
    <td class="tg-pm97"><span style="background-color:#FCFCFF">23.56%</span></td>
    <td class="tg-pm97"><span style="background-color:#FCFCFF">31.40%</span></td>
    <td class="tg-pm97"><span style="background-color:#FCFCFF">26.50%</span></td>
    <td class="tg-pm97"><span style="background-color:#FCFCFF">25.76%</span></td>
    <td class="tg-pm97"><span style="background-color:#FCFCFF">6.19%</span></td>
  </tr>
  <tr>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">Marine Tree</td>
    <td class="tg-0pky">Mod-2.9</td>
    <td class="tg-0pky">HD-CapsNet Without Skip   Connection</td>
    <td class="tg-0pky">5.97</td>
    <td class="tg-x1e6"><span style="background-color:#F8696B">86.98%</span></td>
    <td class="tg-pm97"><span style="background-color:#FCFCFF">77.82%</span></td>
    <td class="tg-pm97"><span style="background-color:#FCFCFF">55.04%</span></td>
    <td class="tg-pm97"><span style="background-color:#FCFCFF">73.35%</span></td>
    <td class="tg-pm97"><span style="background-color:#FCFCFF">75.76%</span></td>
    <td class="tg-pm97"><span style="background-color:#FCFCFF">74.36%</span></td>
    <td class="tg-x1e6"><span style="background-color:#F8696B">86.95%</span></td>
    <td class="tg-x1e6"><span style="background-color:#F8696B">49.34%</span></td>
  </tr>
  <tr>
    <td class="tg-0pky">Marine Tree</td>
    <td class="tg-0pky">Mod-3.3</td>
    <td class="tg-0pky">HD-CapsNet With Consistency Loss   (Lc)</td>
    <td class="tg-0pky">13.58</td>
    <td class="tg-sfj8"><span style="background-color:#63BE7B">89.88%</span></td>
    <td class="tg-sfj8"><span style="background-color:#63BE7B">78.60%</span></td>
    <td class="tg-sfj8"><span style="background-color:#63BE7B">57.15%</span></td>
    <td class="tg-sfj8"><span style="background-color:#63BE7B">75.02%</span></td>
    <td class="tg-sfj8"><span style="background-color:#63BE7B">76.04%</span></td>
    <td class="tg-sfj8"><span style="background-color:#63BE7B">75.44%</span></td>
    <td class="tg-sfj8"><span style="background-color:#63BE7B">94.47%</span></td>
    <td class="tg-sfj8"><span style="background-color:#63BE7B">55.59%</span></td>
  </tr>
  <tr>
    <td class="tg-0pky">Marine Tree</td>
    <td class="tg-0pky">Mod-3.4</td>
    <td class="tg-0pky">HD-CapsNet Without Consistency   Loss (Lc)</td>
    <td class="tg-0pky">13.58</td>
    <td class="tg-pm97"><span style="background-color:#FCFCFF">89.50%</span></td>
    <td class="tg-x1e6"><span style="background-color:#F8696B">77.57%</span></td>
    <td class="tg-x1e6"><span style="background-color:#F8696B">53.75%</span></td>
    <td class="tg-x1e6"><span style="background-color:#F8696B">73.29%</span></td>
    <td class="tg-x1e6"><span style="background-color:#F8696B">74.76%</span></td>
    <td class="tg-x1e6"><span style="background-color:#F8696B">73.88%</span></td>
    <td class="tg-pm97"><span style="background-color:#FCFCFF">92.37%</span></td>
    <td class="tg-pm97"><span style="background-color:#FCFCFF">51.85%</span></td>
  </tr>
</tbody>
</table>