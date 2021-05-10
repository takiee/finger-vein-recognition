# Research and Implementation of Finger Vein Recognition Algorithm
手指静脉识别算法研究与实现
## 工作简述
本文对创新设计课程的整个过程进行了梳理和总计。首先对几种常见的生物特征识别技术进行了概述和比较，重点关注手指静脉识别技术。对指静脉识别图像采集、预处理、特征提取和特征匹配四个阶段都做出了比较详细的分析。由于血液中的血红蛋白可吸收近红外光，因此可采用光反射法或光透视法进行静脉图像的采集。预处理先要获取 ROI 并利用 CLAHE 增强图像，通过多尺度匹配滤波提取静脉纹路并通过 Zhang-Suen 算法细化静脉纹路。本文介绍了四种特征提取与匹配的方法，分别是 LBP 特征提取与匹配、PBBM 特征提取与匹配、基于细节点和 Hausdorff 距离的特征提取与匹配、基于细节点和结构的特征提取与匹配。
## 数据集介绍
使用香港理工大学的手指图像数据库（POLYU），该数据库在 2009 年 4 月到 2010 年 3 月期间使用香港理工大学校园中的非接触式成像设备进行了大规模采集，所有图像均为位图（*.bmp）格式。在此数据集中，约有 93%的受试者年龄小于 30 岁。手指图像是单独采集获取的，最小间隔为一个月，最大间隔为 6 个月以上，平均间隔 66.8天。在每次采集中，每个受试者分别提供食指的六个样本。
## 图像预处理
- 获取ROI  
![image](https://user-images.githubusercontent.com/83262562/117675023-c885cc80-b1de-11eb-827b-eeb1221c05cb.png)
- 图像增强  
![image](https://user-images.githubusercontent.com/83262562/117675122-e0f5e700-b1de-11eb-8837-e37d26fde12a.png)

- 静脉纹路的提取：多尺度匹配滤波  
![image](https://user-images.githubusercontent.com/83262562/117675404-1bf81a80-b1df-11eb-8249-7d7a49184509.png)


- 静脉纹路的细化：Zhang-Suen 细化算法  
![image](https://user-images.githubusercontent.com/83262562/117675254-fc60f200-b1de-11eb-8688-5afe4e17b570.png)

- 静脉纹路的毛刺去除  
![image](https://user-images.githubusercontent.com/83262562/117675281-0256d300-b1df-11eb-9772-fb4e9f0f35cb.png)

## 特征提取和匹配
- 基于LBP的特征提取和匹配
- 基于PBBM的特征提取和匹配
- 基于细节点和MHD的特征提取和匹配
- 基于细节点和结构的特征提取和匹配
## 实验结果展示
### 匹配准确率
算法|准确率
--|--
LBP|72.6%
PBBM|88.9%
MHD|73.5%
### 界面
初始界面
![image](https://user-images.githubusercontent.com/83262562/117674620-69c05300-b1de-11eb-8af4-132e0cc74e19.png)
结果界面
![image](https://user-images.githubusercontent.com/83262562/117674676-75137e80-b1de-11eb-94a5-762e2f86d1b3.png)

