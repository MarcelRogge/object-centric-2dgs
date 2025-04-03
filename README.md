# Object-Centric 2D Gaussian Splatting
[[Website](https://av.dfki.de/publications/object-centric-2d-gaussian-splatting-background-removal-and-occlusion-aware-pruning-for-compact-object-models/)] [[Publication](https://www.scitepress.org/PublishedPapers/2025/133055/)] [[Arxiv](https://arxiv.org/abs/2501.08174)]
## Preamble

Unfortunately, we decided against publishing the code as the specific modifications made are not significant enough to justify hosting a separate repository.
However, we are happy to provide implementation details to make it easier to reproduce our contributions.

Please let us know if there are any parts that are not clear. We would be happy to improve this document.

## Implementation details

Our implementation builds on 2D Gaussian Splatting, the work from Binbin Huang. [[Project Page](https://surfsplatting.github.io)] [[GitHub](https://github.com/hbb1/2d-gaussian-splatting)]<br>
Specifically, the last commit at the time of clone was on Aug 29, 2024. SHA: [19eb5f1](https://github.com/hbb1/2d-gaussian-splatting/commit/19eb5f1e091a582e911b4282fe2832bac4c89f0f)

### Object-Centric Reconstruction

- Handle loading of masks
- [Optionally] Skip training with all empty masks (wasteful & error metrics undefined)
- Add masking for photometric loss
    - See Eq.2 of our paper
    - I.e. 'gt_image * object_mask' & 'image * object_mask' before input into original function
- Add background loss
    - Expose alphas from rendering to train.py
    - Invert mask (1-object_mask &#8594; bg_mask)
    - Compute background loss: loss_bg = average(alphas * bg_mask), see Eq.1 of our paper
    - Add to total loss: total_loss += lambda_bg * loss_bg, see Eq.3 of our paper
    - We set lambda_bg as 0.5 by default, see Section 5.1 of our paper
- [Optionally] Add use of masked error metrics in logging and evaluation

### Occlusion-aware Pruning

#### Integration into the Gaussian Model (./scene/gaussian_model.py)

- Add seen status attribute to Gaussians
    - Function: \_\_init\_\_()
        - self.seen = torch.empty(0)
- Add handling of new attribute in other functions
    - Function: training_setup()
        - self.seen = torch.ones(self.get_xyz.shape[0], device="cuda", dtype=torch.bool, requires_grad=False)
    - Function: prune_points()
        - self.seen = self.seen[valid_points_mask]
    - Function: densification_postfix()
        - self.seen = torch.cat((self.seen, torch.ones(new_opacities.shape[0], device="cuda", dtype=torch.bool, requires_grad=False)), dim=0)
- Add functions to handle adding and clearing seen status
    - Function: e.g. add_seen_status()
        - self.seen[update_filter] = True
    - Function: e.g. clear_seen_status()
        - self.seen = torch.zeros(self.seen.shape[0], device="cuda", dtype=torch.bool, requires_grad=False)
- Add function to prune based on seen status
    - Function: e.g. prune_unseen()
    - if self.seen.sum() < self.seen.numel():<br>
      self.prune_points(~self.seen)
    - self.clear_seen_status()

#### Integration into Adaptive Densification Control (./train.py)

- [Optionally] Automatically decide pruning interval based on number of training cameras
    - Every camera should be seen at least once. We recommend 2 times the number of cameras
- Add seen status from rendering
    - Before original densification
    - gaussians.add_seen_status(render_pkg["seen"])
- Add pruning of occluded Gaussians
    - After original densification
    - if iteration % prune_unseen_interval == 0:<br>
    gaussians.prune_unseen()
- [Optionally] Add pruning step for occluded Gaussians after training loop
    - Clear seen status
    - Render all training cameras once while tracking seen status
    - Prune occluded Gaussians

#### Tracking during Rasterization (CUDA)

- Initialize tensor for tracking seen status (mirroring of 'radii' tracking for visibility from original 2DGS/3DGS)
    - File: ./submodules/diff-surfel-rasterization/rasterize_points.cu
        - Function: RasterizeGaussiansCUDA()
            - torch::Tensor seen = torch::full({P}, 0, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));
- As Input: hand through for updating (until render function of forward.cu)
    - ./submodules/diff-surfel-rasterization/rasterize_points.cu
    - ./submodules/diff-surfel-rasterization/cuda_rasterizer/rasterizer.h
    - ./submodules/diff-surfel-rasterization/cuda_rasterizer/rasterizer_impl.cu
    - ./submodules/diff-surfel-rasterization/cuda_rasterizer/forward.h
    - ./submodules/diff-surfel-rasterization/cuda_rasterizer/forward.cu
- Track seen status during rendering
    - File: ./submodules/diff-surfel-rasterization/cuda_rasterizer/forward.cu
        - Function: renderCUDA()
            - After adding color (Eq.3 from 3DGS paper)
            - seen[collected_id[j]] = true;
- As Output: hand through for densification control (until train.py)
    - ./submodules/diff-surfel-rasterization/rasterize_points.h
    - ./submodules/diff-surfel-rasterization/rasterize_points.cu
    - ./submodules/diff-surfel-rasterization/diff_surfel_rasterization/\_\_init\_\_.py
    - ./gaussian_renderer/\_\_init\_\_.py
    - ./train.py

## Citation
If you use our work, please consider citing it:
```bibtex
@conference{RoggeOC2DGS2025,
    author={Marcel Rogge and Didier Stricker},
    title={Object-Centric 2D Gaussian Splatting: Background Removal and Occlusion-Aware Pruning for Compact Object Models},
    booktitle={Proceedings of the 14th International Conference on Pattern Recognition Applications and Methods - ICPRAM},
    year={2025},
    pages={519-530},
    publisher={SciTePress},
    organization={INSTICC},
    doi={10.5220/0013305500003905},
    isbn={978-989-758-730-6},
    issn={2184-4313}
}
```