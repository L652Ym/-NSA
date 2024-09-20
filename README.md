<div align="center">
<h1>
  StrongSort-Pip: Packaged version of StrongSort 
</h1>
<h4>
    <img width="700" alt="teaser" src="docs/uav.gif">
</h4>
<div>
    <a href="https://pepy.tech/project/strongsort"><img src="https://pepy.tech/badge/strongsort" alt="downloads"></a>
    <a href="https://badge.fury.io/py/strongsort"><img src="https://badge.fury.io/py/strongsort.svg" alt="pypi version"></a>
</div>
</div>

## <div align="center">Overview</div>

This repo is a packaged version of the [StrongSort](https://github.com/dyhBUPT/StrongSORT) algorithm.

### Installation

```
pip install strongsort
```

### Detection Model + StrongSort

```python
from strongsort import StrongSORT

tracker = StrongSORT(model_weights='model.pth', device='cuda')
pred = model(img)
for i, dets in enumerate(pred):
    # dets: [x1, y1, x2, y2, confidence, class]
    tracks = tracker.update(dets, ori_img=frame)
```

## Citations

```bibtex
@article{du2022strongsort,
  title={Strongsort: Make deepsort great again},
  author={Du, Yunhao and Song, Yang and Yang, Bo and Zhao, Yanyun},
  journal={arXiv preprint arXiv:2202.13514},
  year={2022}
}
```
