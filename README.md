# A consistency-aware deep capsule network for hierarchical multi-label image classification

This is the official implementation of the paper titled "A consistency-aware deep capsule network for hierarchical multi-label image classification" by Khondaker Tasrif Noor, Antonio Robles-Kelly, Leo Yu Zhang, Mohamed Reda Bouadjenek, and Wei Luo. The paper is available on [Neurocomputing Journal.](https://doi.org/10.1016/j.neucom.2024.128376)

# Abstract

Hierarchical classification is a significant challenge in computer vision due to the logical order and interconnectedness of multiple labels. This paper presents HD-CapsNet, a novel neural network architecture based on deep capsule networks, specifically designed for hierarchical multi-label classification(HMC). By incorporating a tree-like hierarchical structure, HD-CapsNet is designed to leverage the inherent ontological order within the hierarchical label tree, thereby ensuring classification consistency across different levels. Additionally, we introduce a specialized loss function that promotes accurate hierarchical relationships while penalizing inconsistencies. This not only enhances classification performance but also strengthens the network’s robustness. We rigorously evaluate HD-CapsNet’s efficacy by benchmarking it against existing HMC methods across six diverse datasets: Fashion-MNIST, Marine-Tree, CIFAR-10, CIFAR-100, Caltech-UCSD Birds-200-2011, and Stanford Cars. Our results conclusively demonstrate that HD-CapsNet excels in learning hierarchical relationships and significantly outperforms the competition in various image classification tasks.

# Results:

---

| Dataset       | Models                  | Total  Trainable params (M) | Accuracy Coarse | Accuracy Medium | Accuracy Fine | Hierarchical Precision | Hierarchical Recall | Hierarchical F1-Score | Consistency | Exact Match |
| ------------- | ----------------------- | --------------------------- | --------------- | --------------- | ------------- | ---------------------- | ------------------- | --------------------- | ----------- | ----------- |
|               |                         |                             |                 |                 |               |                        |                     |                       |             |             |
| Fashion MNIST | HD-CapsNet              | 4.82                        | 99.92%          | 97.79%          | 94.83%        | 97.51%                 | 97.54%              | 97.52%                | 99.84%      | 94.74%      |
| Fashion MNIST | HD-CapsNet $~\dagger$  | 4.82                        | 99.89%          | 97.78%          | 94.92%        | 97.53%                 | 97.59%              | 97.55%                | 99.70%      | 94.77%      |
| Fashion MNIST | HD-CapsNet $~\ddagger$ | 4.73                        | 99.91%          | 97.63%          | 94.66%        | 97.40%                 | 97.42%              | 97.41%                | 99.87%      | 94.60%      |
|               |                         |                             |                 |                 |               |                        |                     |                       |             |             |
| CIFAR-10      | HD-CapsNet              | 5.23                        | 98.79%          | 94.28%          | 91.22%        | 94.74%                 | 94.89%              | 94.80%                | 99.18%      | 90.95%      |
| CIFAR-10      | HD-CapsNet $~\dagger$  | 5.23                        | 98.71%          | 94.01%          | 90.97%        | 94.53%                 | 94.73%              | 94.62%                | 98.99%      | 90.58%      |
| CIFAR-10      | HD-CapsNet $~\ddagger$ | 4.84                        | 98.76%          | 93.36%          | 90.26%        | 94.09%                 | 94.30%              | 94.18%                | 98.94%      | 89.85%      |
|               |                         |                             |                 |                 |               |                        |                     |                       |             |             |
| CIFAR-100     | HD-CapsNet              | 7.85                        | 86.93%          | 79.31%          | 66.38%        | 77.43%                 | 79.20%              | 78.12%                | 89.80%      | 63.80%      |
| CIFAR-100     | HD-CapsNet $~\dagger$  | 7.85                        | 86.81%          | 78.73%          | 66.23%        | 77.10%                 | 79.02%              | 77.85%                | 88.62%      | 63.36%      |
| CIFAR-100     | HD-CapsNet $~\ddagger$ | 5.55                        | 86.57%          | 78.33%          | 57.08%        | 73.86%                 | 75.00%              | 74.31%                | 92.51%      | 56.10%      |
|               |                         |                             |                 |                 |               |                        |                     |                       |             |             |
| Marine Tree   | HD-CapsNet              | 13.58                       | 89.88%          | 78.60%          | 57.15%        | 75.02%                 | 76.04%              | 75.44%                | 94.47%      | 55.59%      |
| Marine Tree   | HD-CapsNet $~\dagger$  | 13.58                       | 89.50%          | 77.57%          | 53.75%        | 73.29%                 | 74.76%              | 73.88%                | 92.37%      | 51.85%      |
| Marine Tree   | HD-CapsNet $~\ddagger$ | 5.97                        | 86.98%          | 77.82%          | 55.04%        | 73.35%                 | 75.76%              | 74.36%                | 86.95%      | 49.34%      |
|               |                         |                             |                 |                 |               |                        |                     |                       |             |             |
| CU Bird       | HD-CapsNet              | 106.01                      | 40.42%          | 21.61%          | 13.39%        | 23.47%                 | 30.33%              | 26.01%                | 27.34%      | 8.63%       |
| CU Bird       | HD-CapsNet $~\dagger$  | 106.01                      | 36.59%          | 17.78%          | 10.87%        | 20.29%                 | 26.56%              | 22.62%                | 24.09%      | 6.28%       |
| CU Bird       | HD-CapsNet $~\ddagger$ | 47.56                       | 35.66%          | 16.98%          | 2.14%         | 14.97%                 | 20.86%              | 17.13%                | 21.44%      | 1.55%       |
|               |                         |                             |                 |                 |               |                        |                     |                       |             |             |
| Stanford Cars | HD-CapsNet              | 81.17                       | 53.34%          | 19.52%          | 14.05%        | 26.73%                 | 34.69%              | 29.73%                | 29.15%      | 8.13%       |
| Stanford Cars | HD-CapsNet $~\dagger$  | 81.17                       | 47.50%          | 16.39%          | 11.74%        | 23.56%                 | 31.40%              | 26.50%                | 25.76%      | 6.19%       |
| Stanford Cars | HD-CapsNet $~\ddagger$ | 25.85                       | 46.01%          | 12.29%          | 1.57%         | 17.10%                 | 24.04%              | 19.79%                | 13.60%      | 0.87%       |

**HERE,** $\dagger$ denotes the HD-CapsNet models without the proposed consistency loss (Lc), and $\ddagger$ denotes those without the skip connections between the secondary capsule layers, respectively.

# Citation

If you find this repository useful for your research or if it helps in your project, please consider citing it.

```
Noor, K.T., Robles-Kelly, A., Zhang, L.Y., Bouadjenek, M.R., Luo, W., 2024. A consistency-aware deep capsule network for hierarchical multi-label image classification. Neurocomputing 604, 128376. https://doi.org/10.1016/j.neucom.2024.128376
```

#### Bibtex Citation Style:

```bash
 @article{Noor_Robles-Kelly_Zhang_Bouadjenek_Luo_2024, title={A consistency-aware deep capsule network for hierarchical multi-label image classification}, volume={604}, ISSN={0925-2312}, DOI={10.1016/j.neucom.2024.128376}, abstractNote={Hierarchical classification is a significant challenge in computer vision due to the logical order and interconnectedness of multiple labels. This paper presents HD-CapsNet, a novel neural network architecture based on deep capsule networks, specifically designed for hierarchical multi-label classification(HMC). By incorporating a tree-like hierarchical structure, HD-CapsNet is designed to leverage the inherent ontological order within the hierarchical label tree, thereby ensuring classification consistency across different levels. Additionally, we introduce a specialized loss function that promotes accurate hierarchical relationships while penalizing inconsistencies. This not only enhances classification performance but also strengthens the network’s robustness. We rigorously evaluate HD-CapsNet’s efficacy by benchmarking it against existing HMC methods across six diverse datasets: Fashion-MNIST, Marine-Tree, CIFAR-10, CIFAR-100, Caltech-UCSD Birds-200-2011, and Stanford Cars. Our results conclusively demonstrate that HD-CapsNet excels in learning hierarchical relationships and significantly outperforms the competition in various image classification tasks. Our implementation is available at https://github.com/tasrif-khondaker/HD-CapsNet.}, journal={Neurocomputing}, author={Noor, Khondaker Tasrif and Robles-Kelly, Antonio and Zhang, Leo Yu and Bouadjenek, Mohamed Reda and Luo, Wei}, year={2024}, month=nov, pages={128376}, language={en-GB} }


```
