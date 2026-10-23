# Third-party components

| Component | Origin | Distribution / terms |
|---|---|---|
| ConvNeXt study model, physical window selection, anatomical rendering and orientation logic | [Heliosli](https://www.kaggle.com/code/heliosli/0-957-latest-dataset-score-in-one-click), including the code redistributed in [nartaa's public package](https://www.kaggle.com/datasets/nartaa/rsna-knee-s6fall-weights) | Adapted source; upstream code identified as Apache-2.0 in that package's README |
| `public_convnext_small.pt` | nartaa, original file `E_prior_s42_fall.pt` | CC BY-NC 4.0; unchanged bytes; credit nartaa and Heliosli |
| `anatomy.pt` | SKM-TEA VNet as distributed by [Heliosli](https://www.kaggle.com/datasets/heliosli/rsna-orthofoundation-fold0-epoch7-20261004) | Downloaded from the original dataset; not relicensed or rehosted in this release |
| `side_cls.pt` | [Heliosli's orientation model](https://www.kaggle.com/datasets/heliosli/rsna-knee-compact-v1) | Downloaded from the original dataset; not relicensed or rehosted in this release |
| VNet transitions in `knee_mri/preprocessing/vnet.py` | MONAI Consortium, MONAI 0.6 | Apache-2.0; copyright header retained; layer factories replaced by equivalent PyTorch layers |
| ConvNeXt and ImageNet initialization | [timm](https://github.com/huggingface/pytorch-image-models), [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) | Dependencies downloaded separately; their upstream licenses apply |

The public checkpoint's original SHA-256 is `0c7297d58f061ab4e5d7c55046f51fa3e8934928046f0647a0c7854e98371c12`. Its license is [Creative Commons Attribution–NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/). The Apache-2.0 license on this repository does not override it.

The preprocessing source has been packaged as importable modules, with directory-based DICOM access and an explicit command-line interface. Unused report-state heads, unrelated model families, experiment launchers, source-path patches and partial-result fallbacks have been removed. The anatomy VNet retains the original parameter names and operations while depending directly on PyTorch.

No competition images, clinical reports, annotation tables, patient identifiers, access credentials or upstream training data are included.

If Kaggle requires authentication for an auxiliary file, download only the named file through the Kaggle CLI and place it in `weights/`:

```bash
kaggle datasets download -d heliosli/rsna-orthofoundation-fold0-epoch7-20261004 -f anatomy.pt -p weights
kaggle datasets download -d heliosli/rsna-knee-compact-v1 -f side_cls.pt -p weights
```

Extract the named `.pt` file if Kaggle returns a ZIP. `knee-mri download` will verify the existing file and skip it.
