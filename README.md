# Fast-rPPG
NoobHeart model implemented with numpy, which can be used for embedded devices.  

## Model
The model is trained on the RLAP dataset using the TensorFlow framework, a total of 169 trainable parameters are stored in float32, then all weights are extracted and the network architecture is re-implemented with numpy. For a detailed evaluation of this model, please refer to [PhysBench](https://github.com/KegangWangCCNU/PhysBench).  
## Additional processing  
This code does not currently include preprocessing and postprocessing parts. It is recommended to use OpenCV for face detection, and scale it to 8x8 resolution using area average sampling (cv2.INTER_AREA). The model uses (Batch, Depth, 8, 8, 3) RGB input and outputs (Batch, Depth) BVP waveforms. This code does not currently include filters and heart rate extraction algorithms.

## Computational overhead  
The computational cost of this algorithm can be ignored, however, due to the large number of loop operations in manual 3D CNN, it is limited by the performance of the Python interpreter and there will be a certain delay. One solution is to use `numba.jit` to speed up loops, with 32 frames input, the delay can be reduced to 1ms. Rewriting this algorithm in C++ is also a solution with similar performance as `numba.jit`.
