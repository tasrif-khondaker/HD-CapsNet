# Modification of HD-CapsNet
 Hierarchical Deep Capsule Network For Image Classification. The model employs capsule network for each hierarchical levels, treating as a global classification model.
 
 **The original architecture of the HD-CapsNet is as follows:**
 
 ![fig:Architecture](Results/Architecture.png?raw=true "Architecture of the HD-CapsNet")
 
## The following changes are made in the architecture.
- Changing the Secondary Capsule Dimension:
    1. 8D>12D>16D (Coarse>Medium>FINE)
		1.1 Applying top-down approach (Coarse-to-Fine)
		1.2 Applying bottom-up approach (Fine-to-Coarse)
### For both (1.1 and 1.2) train the model with and without the consistency loss $L_{cons}$ function
	**For training the model without $L_{cons}$ just applied Hinge loss (Margin Loss) for each level**

#Results:
