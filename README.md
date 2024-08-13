# MRM: radar dataset for micro-motion targets.

> **Warning**❗:  This repository is still under construction, as the corresponding paper is still under submission. The uploaded code is not yet complete, and we are still performing final validation and testing. The dataset upload is ongoing, and we are currently searching for a suitable dataport, as we require 2TB of storage. We welcome any related inquiries.
>
> TODO: 
>
> - add rotated figure.
> - fix plot ticks.
> - Electromagnetic simulation part.
> - experiment example guide.
> - code validation & upload.
> - find dataport

## Introduction

The MRM dataset is a radar dataset generated through simulation. It features a scenario where a fully-polarized ISAR radar is used to track and image space micro-motion targets for classification. Consequently, it includes wideband and narrowband radar echoes under full polarization, with an observation time of 2 seconds.

The purpose of creating this dataset is straightforward: existing research does not make their datasets publicly available, and it is challenging to obtain these datasets by contacting the authors of the relevant papers. Of course, it is possible to perform simulations independently, but the details of the simulations can lead to significant differences in the data, making meaningful comparisons impossible. Moreover, electromagnetic simulations are highly time-consuming, and we don't want any researchers to waste their time on these tasks.

This dataset (and the additional versions it includes) can be applied to a variety of tasks. Here, I will only list some of the prior research; for detailed examples, please refer to these papers.

- Create new models for radar target recognition. Related work includes:[[1]](Recognition of Micro-Motion Space Targets Based on Attention-Augmented Cross-Modal Feature Fusion Recognition Network), [[2]](https://ieeexplore.ieee.org/document/9691916/), [[3]](https://ieeexplore.ieee.org/document/10281917/), [[4]](https://ieeexplore.ieee.org/document/10431715/). 
- Generate data using a small amount of existing data to improve classification accuracy (since such observational data is difficult to obtain in real-world scenarios). Related work includes: [[5]](https://www.mdpi.com/2072-4292/15/21/5085).

- Micro-motion parameter estimation. Related work includes: [[6]](https://www.mdpi.com/2072-4292/14/15/3691).

Of course, these research directions are just broad summaries. Various detailed studies, such as exploring effective feature fusion methods, investigating polarization data fusion for recognition, extracting micro-motion curves, and experimenting with different radar signal processing techniques, can all make use of this dataset.

The structure of the data in the dataset and its application methods will be explained with simple guides and code examples in the following sections. For more details, please refer to the full text of the paper.

## Kinematic model

To simplify the question, we only use reference coordinates to describe the earth, the orbit of target, the reference coordinates of radar, and the target body axis, which means we lack the geoscience description, but they are sufficient to describe the procession of a micro-motion target movement.

Firstly, the orbit of target is only decided by the release velocity $v$ and position $r$, which means we can use them to calculate orbit elements: 

- $a$
- $e$
- $i$
- $\Omega$
- $\omega$
- $f$

With those elements and the position of release, the flight procession can be plot as follow.

<table>
    <tr>
        <td style="text-align: center;">
            <img src="./figure/animation_0.gif" alt="Image 1" width="500"/>
            <br>
            <i>viewing angle: elev=0,azim=90.</i>
        </td>
        <td style="text-align: center;">
            <img src="./figure/animation_1.gif" alt="Image 2" width="500"/>
            <br>
            <i>viewing angle: elev=-65,azim=63.</i>
        </td>
    </tr>
</table>

Now, let's consider the attitude of the target. Taking the conical target in the figure below as an example, we assume that the re-entry direction of the target is fixed (as designed), with its head pointing towards the ground, which means the warhead should be oriented below the y-axis (with the orbit in the $YOZ$ plane). Due to spin stabilization, the precession axis will remain fixed in inertial space. (TODO: add rotated figure.)

<table>
    <tr>
        <td style="text-align: center;">
            <img src="./figure/output.png" alt="Image 1" width="300"/>
            <br>
            <i>Origin</i>
        </td>
        <td style="text-align: center;">
            <img src="./figure/output.png" alt="Image 2" width="300"/>
            <br>
            <i>Rotated</i>
        </td>
    </tr>
</table>

Adding micro-motion to the existing attitude results in the final model.

<table>
    <tr>
        <td style="text-align: center;">
            <img src="./figure/30130.gif" alt="Image 1" width="300"/>
            <br>
            <i>Coning</i>
        </td>
        <td style="text-align: center;">
            <img src="./figure/090.gif" alt="Image 2" width="300"/>
            <br>
            <i>Coning</i>
        </td>
        <td style="text-align: center;">
            <img src="./figure/test.gif" alt="Image 2" width="300"/>
            <br>
            <i>Nutation</i>
        </td>
    </tr>
</table>



## Electromagnetic simulation

