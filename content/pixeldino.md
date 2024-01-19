+++
title = "PixelDINO"
draft = false
+++

{{< figure src="rts.jpg" caption=" " >}}

Retrogressive Thaw Slumps are a permafrost disturbance
comparable to landslides induced by permafrost thaw.
Their detection and monitoring is important for understanding
the dynamics of permafrost thaw and the vulnerability of permafrost across the Arctic.
To do this efficiently with deep learning,
large amounts of annotated data are needed,
of which currently we do not have enough.

{{< figure src="map.png" caption=" ">}}

In order to address this without needing to manually digitize
vast areas across the Arctic,
we propose a semi-supervised learning approach which is able to combine
existing labelled data with additional unlabelled data.

{{< figure src="pixeldino.png" caption=" ">}}

This is done by asking the model to derive pseudo-classes,
according to which it will segment the unlabelled images.
For these pseudo-classes,
consistency across data augmentations is enforced,
which provides valuable training feedback to the model even for unlabelled tiles.

