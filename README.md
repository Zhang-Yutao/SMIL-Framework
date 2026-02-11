# CVS Assessment via Distillation-based Self-Supervised and Multiple Instance Learning in Laparoscopic Cholecystectomy
This is the code base for paper [CVS Assessment via Distillation-based Self-Supervised and Multiple Instance Learning in Laparoscopic Cholecystectomy](https://doi.org/10.1007/s11548-026-03580-9), which has been accepted by the International Journal of Computer Assisted Radiology and Surgery.

# Install and compile the prerequisites
* Python 3.8
* PyTorch >= 1.8
* NVIDIA GPU + CUDA
* Python packages: numpy, opencv-python, scipy

# Pretrained model
Once the article is officially published, our weights will be updated accordingly.

# Main experiment
1. Modify the data path in trainlist.txt and testlist.txt to your own data path.
2. Run [python SMIL.py].

## Citation
```
@article{wang2026smil,
  title = {CVS Assessment via Distillation-based Self-Supervised and Multiple Instance Learning in Laparoscopic Cholecystectomy},
  shorttitle = {SMIL},
  author = {Hao, Wang and Yutao, Zhang and Yuxuan, Yang and Yuanbo, Zhu and Rui, Xu},
  year={2026},
  doi={10.1007/s11548-026-03580-9},
  journal = {International Journal of Computer Assisted Radiology and Surgery}
}
```