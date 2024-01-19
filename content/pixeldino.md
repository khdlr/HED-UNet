+++
title = "HED-UNet"
draft = false
+++

{{< figure src="highlevel.png" caption=" " >}}

HED-UNet combines the commonly used semantic segmentation model UNet with the edge detection model HED to reflect the need to focus on the edges.

{{< figure src="hed-unet.png" caption=" " >}}

For allowing the model to use a larger spatial context and "use a bigger brush", the number of up- and downsampling layers was increased from 4 to 6.
Finally, to retain fine-grained details near the edges but use broader contextual information farther away from the edges, the predictions at different resolution levels are merged through an attention merging head.
