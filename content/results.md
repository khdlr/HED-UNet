+++
title = "Results"
draft = false
+++

{{< figure src="predictions.jpg" caption=" ">}}

As a multi-task model, HED-UNet predicts both segmentation maps and edge detection maps.
Compared to state-of-the-art models for both these tasks, it shows superior performance when applied for calving front detection.

{{< figure src="attnmap.png" caption=" ">}}

Inspecting the attention maps used for merging information from different resolutions shows that the model is indeed behaving as expected: High-resolution information is used in the boundary areas, while the more robust, low-resolution information is used in other regions.
