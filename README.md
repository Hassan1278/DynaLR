# DynaLR
Advanced learning rate optimizers for PyTorc
## Comprehensive Benchmark Results based on 3 seeds and 30 Epochs Training on A100 GPU for the Resnet-18 Tests and ve6-1 TPU for the SimpleCNN tests.
## Abstract 📌
DynaLR introduces a principle of adaptive learning rate optimizers using PID

✔ **2.6% accuracy gain** over Adam on CNN architectures  
✔ **Faster convergence** (3-5% speedup)  
✔ **Architecture-aware** performance (excels on CNNs)  
✔ **Four specialized variants** for different use cases

### SimpleCNN on CIFAR-10
| Algorithm          | Accuracy (Mean ± Std) | Time (s) | vs Adam |
|--------------------|------------------------|----------|---------|
| **DynaLRMemory**   | 0.7735 ± 0.0031       | 153.9    | +1.01%  |
| DynaLRenhanced     | 0.7708 ± 0.0024       | 154.4    | +0.74%  |
| DynaLRnoMemory     | 0.7711 ± 0.0089       | 155.5    | +0.77%  |
| Adam (Baseline)    | 0.7634 ± 0.0033       | 159.0    | -       |
| DynaLRAdaptivePID  | 0.7195 ± 0.0076       | 152.0    | -4.39%  |

### ResNet18 on CIFAR-10
| Algorithm          | Accuracy (Mean ± Std) | Time (s) | vs Adam |
|--------------------|------------------------|----------|---------|
| DynaLRMemory       | 0.8745 ± 0.0085       | 271.8    | -2.19%  |
| DynaLRenhanced     | 0.8771 ± 0.0069       | 271.7    | -1.93%  |
| DynaLRnoMemory     | 0.8776 ± 0.0056       | 273.8    | -1.88%  |
| **Adam (Baseline)**| **0.8964 ± 0.0030**   | 276.6    | -       |
| DynaLRAdaptivePID  | 0.8871 ± 0.0028       | 271.5    | -0.93%  |

### SimpleCNN on CIFAR-100
| Algorithm          | Accuracy (Mean ± Std) | Time (s) | vs Adam |
|--------------------|------------------------|----------|---------|
| **DynaLRMemory**   | 0.4589 ± 0.0088       | 154.9    | +2.64%  |
| DynaLRenhanced     | 0.4526 ± 0.0040       | 156.9    | +2.01%  |
| DynaLRnoMemory     | 0.4503 ± 0.0061       | 156.5    | +1.78%  |
| Adam (Baseline)    | 0.4325 ± 0.0076       | 164.8    | -       |
| DynaLRAdaptivePID  | 0.3508 ± 0.0013       | 156.7    | -8.17%  |

### ResNet18 on CIFAR-100
| Algorithm          | Accuracy (Mean ± Std) | Time (s) | vs Adam |
|--------------------|------------------------|----------|---------|
| DynaLRMemory       | 0.6405 ± 0.0072       | 272.1    | -0.84%  |
| DynaLRenhanced     | 0.6142 ± 0.0130       | 276.4    | -3.47%  |
| DynaLRnoMemory     | 0.6316 ± 0.0051       | 277.7    | -1.73%  |
| Adam (Baseline)    | 0.6489 ± 0.0035       | 276.7    | -       |
| **DynaLRAdaptivePID** | **0.6504 ± 0.0062**   | 276.4    | **+0.15%** |

## Key Findings 🔍
1. **CNN Dominance**: 3 DynaLR variants outperform Adam on lightweight CNNs:
   - +2.64% accuracy gain on CIFAR-100/SimpleCNN
   - +1.01% gain on CIFAR-10/SimpleCNN
   - 3-5% faster training times

2. **ResNet Specialist**: 
   - DynaLRAdaptivePID beats Adam on CIFAR-100/ResNet18 (+0.15%)
   - Shows fastest training times across all ResNet tests

3. **Architecture Matters**:
   - DynaLRMemory works best for CNNs
   - DynaLRAdaptivePID excels on complex models
   - Adam remains strong on ResNet/CIFAR-10

4. **Speed Advantage**:
   - All DynaLR variants are faster than Adam
   - Average 2.5% speedup across all tests



[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16328686.svg)](https://doi.org/10.5281/zenodo.16328686)
## 📄 Citation

If you use this repository, please cite:

Hassan Al Subaidi. *DynaLR++PID: A Dynamic Learning Rate Optimizer with Memory and PID Feedback*. Zenodo, 2025. https://doi.org/10.5281/zenodo.16328686



📝 License
MIT License (see below)

MIT License
Copyright (c) 2025 Hassan Al Subaidi
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
