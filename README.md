# CVS Assessment via Distillation-based Self-Supervised and Multiple Instance Learning in Laparoscopic Cholecystectomy
This is the code base for paper CVS Assessment via Distillation-based Self-Supervised and Multiple Instance Learning in Laparoscopic Cholecystectomy (International Journal of Computer Assisted Radiology and Surgery, 2025)

Abstract

Due to the high risk of bile duct injuries during laparoscopic cholecystectomy (LC), accurate and automated assessment of the Critical View of Safety (CVS) plays a vital role in assisting surgeons during procedures and reducing the incidence of such injuries. Existing methods predominantly rely on costly semantic segmentation annotations or continuous sequential image inputs, limiting the generalization performance and temporal-spatial understanding capability of the models. This study proposes an efficient automated CVS assessment framework to eliminate the dependency on segmentation annotations, enhance the spatiotemporal comprehension of LC procedures, and improve the generalization and robustness of the evaluation model. We introduce SMIL framework, a novel automated CVS assessment framework that integrates distilled self-supervised and multiple instance learning strategies. Initially, a video transformer is pretrained using Self-Distillation with No Labels on laparoscopic cholecystectomy videos to capture the spatiotemporal features of the surgical procedure. Subsequently, SMIL leverages a multiple instance learning approach, fusing global-level features with local instance-level representations, allowing for end-to-end optimization and accurate CVS classification. Experimental results demonstrate that the proposed SMIL framework significantly outperforms existing state-of-the-art methods. Notably, SMIL surpasses prior methods that rely on pixel-wise segmentation labels, despite requiring no such annotations. Specifically, SMIL framework achieves an improvement of 5.98\% in mean average precision and a 2.74\% increase in balanced accuracy compared to the best-performing baseline model, establishing a new benchmark in automated CVS evaluation. The SMIL framework effectively accomplishes automated CVS assessment without relying on expensive segmentation annotations and alleviates dependency on sequential frame inputs. By innovatively integrating self-supervised learning with multi-instance learning, the model substantially enhances temporal-spatial comprehension and generalization capability in LC surgeries, offering significant theoretical insights and practical value for improving surgical safety assessments.

# Install and compile the prerequisites
* Python 3.8
* PyTorch >= 1.8
* NVIDIA GPU + CUDA
* Python packages: numpy, opencv-python, scipy

# Pretrained model
Once the article is published, our weights will be updated accordingly.

# Main experiment
1. Modify the data path in trainlist.txt and testlist.txt to your own data path.
2. Run [python main_SMIL.py].
