# RDMFuse
Codes of ***A Retinex Decomposition Model-Based Deep Framework for Infrared and Visible Image Fusion. (JSTSP)***

# Abstract
Infrared and visible image fusion (IVIF) aims to integrate complementary information between sensors and generate information-rich high-quality images. However, current methods mainly concentrate on the fusion of the source features from the sensors, ignoring the feature information mismatch caused by the property of the sensors, which results in redundant or even invalid information.   To tackle the above challenges, this paper developed an end-to-end model based on the Retinex Decomposition Model (RDM), called RDMFuse, which utilizes a hierarchical feature process to alleviate the fusion performance degradation caused by the feature-level mismatch. Specifically, as infrared images only provide an overview of the intrinsic properties of the scene, we first use RDM to decouple visible images into a reflectance component containing intrinsic properties and an illumination component containing illumination information. Then, the contrast texture module (CTM) and the intrinsic fusion function are designed for the property of the intrinsic feature, which complements each other to aggregate the intrinsic information of the source images at a smaller cost and brings the fused image more comprehensive scene information. Besides, the illumination-adaptive module implements illumination component optimization in a self-supervised way to make the fused image with an appropriate intensity distribution. It is worth noting that this mechanism implicitly improves the entropy quality of the image to improve the image degradation problem caused by environmental factors, especially in the case of a dark environment. Numerous experiments have demonstrated the effectiveness and robustness of the RDMFuse and the superiority of generalization in high-level vision tasks due to the improved discriminability of the fused image to the captured scene.

# Testing
If you want to infer with our EMMA and obtain the fusion results in our paper, please run ```test.py```.
Then, the fused results will be saved in the ```'./Fused image/'``` folder.

# Training
You can change your own data address in ```dataset.py``` and use ```train.py``` to retrain the method.

# RDMFuse Results

