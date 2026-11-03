---
title: CC-4DGS
emoji: 🎞️
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.50.0
python_version: "3.10"
app_file: app.py
pinned: false
license: other
short_description: Storage-efficient dynamic Gaussian splatting viewer
---

# CC-4DGS viewer (ZeroGPU)

Interactive renderer for **CC-4DGS: Computational Deformation and Point-Cloud
Compression for Storage-Efficient Dynamic Gaussian Splatting** (IEEE TVCG 2026,
[arXiv:2609.02184](https://arxiv.org/abs/2609.02184),
[code](https://github.com/KyungdaePark/CC-4DGS)).

Load a trained scene and render it at any time step from the dataset cameras,
the N3DV spiral trajectory, or an orbit:

- **CDF** checkpoints: `point_cloud/iteration_<N>/{point_cloud.ply, deformation.pth}`
- **CDF+CCA** checkpoints: `point_cloud/iteration_<N>/{compressed/, deformation.pth}`
- optional cameras: `poses_bounds.npy` (N3DV) or COLMAP `images.bin` + `cameras.bin` (Technicolor)

Upload a zip of the scene folder or point the app at a folder in a Hugging Face
model/dataset repo. Set the Space variables `CC4DGS_HUB_REPO`,
`CC4DGS_HUB_SUBFOLDER` and `CC4DGS_HUB_REPO_TYPE` to pre-fill the Hub fields;
set the `HF_TOKEN` secret for private repos.

Only rendering runs on the GPU (`@spaces.GPU`); checkpoint decoding, CCA
decompression and camera setup run on CPU. The rasterizer and `simple-knn`
CUDA extensions are installed at start-up from `wheels/` (sm_80/86/89/90), or
compiled from `submodules/` when the wheels do not match the installed torch.

## License

The CC-4DGS code is Apache-2.0. The Inria Gaussian Splatting rasterizer and
`simple-knn` are restricted to non-commercial research and evaluation use; see
`LICENSE.md` and `THIRD_PARTY_NOTICES.md`.
