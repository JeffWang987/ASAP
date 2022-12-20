<div align="center">   

# Are We Ready for Vision-Centric Driving Streaming Perception? The ASAP Benchmark
</div>

- **Are We Ready for Vision-Centric Driving Streaming Perception? The ASAP Benchmark** [Paper in arXiv](https://arxiv.org/pdf/2212.08914.pdf)

- **The 12Hz nuScenes-H dataset**
<div align="center">  

https://user-images.githubusercontent.com/49095445/208249248-8ac3462e-8440-4982-a4e2-077daa7fac48.mp4

</div>


# Abstract

In recent years, vision-centric perception has flourished in various autonomous driving tasks, including 3D detection, semantic map construction, motion forecasting, and depth estimation. Nevertheless, the latency of vision-centric approaches is too high for practical deployment (e.g., most camera-based 3D detectors have a runtime greater than 300ms). To bridge the gap between ideal research and real-world applications, it is necessary to quantify the trade-off between performance and efficiency. Traditionally, autonomous-driving perception benchmarks perform the **offline** evaluation, neglecting the inference time delay. To mitigate the problem, we propose the **A**utonomous-driving **S**tre**A**ming **P**erception (ASAP) benchmark, which is the first benchmark to evaluate the **online** performance of vision-centric perception in autonomous driving. On the basis of the 2Hz annotated nuScenes dataset, we first propose an annotation-extending pipeline to generate high-frame-rate labels for the 12Hz raw images. Referring to the practical deployment, the **S**treaming **P**erception **U**nder const**R**ained-computation (SPUR) evaluation protocol is further constructed, where the 12Hz inputs are utilized for streaming evaluation under the constraints of different computational resources. In the ASAP benchmark, comprehensive experiment results reveal that the model rank alters under different constraints, suggesting that the model latency and computation budget should be considered as design choices to optimize the practical deployment. To facilitate further research, we establish baselines for camera-based streaming 3D detection, which consistently enhance the streaming performance across various hardware.

- **Streaming perception results**
<div align="center">  

<img width="881" alt="streaming_perception" src="https://user-images.githubusercontent.com/49095445/208249337-f26d2951-499a-4d85-924a-a742f35f7a24.png">

</div>

# Getting Started

- [Installation](docs/install.md) 

- [Prepare Dataset](docs/prepare_data.md)

- [Streaming Eval](docs/streaming_evaluation.md)

  

# Bibtex
If this work is helpful for your research, please consider citing the following BibTeX entry.

```
@article{wang2022asap,
  title={Are We Ready for Vision-Centric Driving Streaming Perception? The ASAP Benchmark},
  author={Wang, Xiaofeng and Zhu, Zheng and Zhang, Yunpeng and Huang, Guan and Ye, Yun and Xu, Wenbo and Chen, Ziwei and Wang, Xingang}
  journal={arXiv preprint arXiv:2212.08914},
  year={2022}
}
```

# Acknowledgement

Many thanks to these excellent projects:
- [sAP](https://github.com/mtli/sAP) 
- [MMDet3D](https://github.com/open-mmlab/mmdetection3d)
- [BEVDet](https://github.com/HuangJunJie2017/BEVDet)
- [PETR](https://github.com/megvii-research/PETR)
- [BEVFormer](https://github.com/fundamentalvision/BEVFormer)
- [BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth)
