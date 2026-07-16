# Third-Party Notices

CC-4DGS builds on open-source research software. The root Apache License 2.0
applies only to original CC-4DGS contributions unless a file states otherwise.
Upstream copyright notices and license terms remain in effect for derived and
third-party files.

## 4DGaussians

- Project: <https://github.com/hustvl/4DGaussians>
- License: Apache License 2.0
- Use in this repository: training, rendering, dataset, and deformation
  infrastructure adapted for CC-4DGS.

## 3D Gaussian Splatting

- Project: <https://github.com/graphdeco-inria/gaussian-splatting>
- Copyright: Inria and the Max Planck Institute for Informatics
- License: Gaussian-Splatting License, restricted to non-commercial research
  and evaluation use.
- Use in this repository: files carrying the Inria copyright header, the
  vendored differentiable Gaussian rasterizer, and the `simple-knn` CUDA
  extension.

The full Gaussian-Splatting License is included at
`submodules/diff-gaussian-rasterization/LICENSE.md`. The `simple-knn` source is
retrieved from the upstream Inria repository at the commit pinned by this
repository's Git submodule entry and retains the license notice in each source
file.

## Other Dependencies

Python and system dependencies installed through `requirements.txt` retain
their respective licenses. Bundled GLM headers under
`submodules/diff-gaussian-rasterization/third_party/glm/` retain the copyright
and license notices included in that directory.

This notice is informational and does not replace or modify any upstream
license.
