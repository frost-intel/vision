import torch
import triton

from tochvision.ops.triton.roi_pool import triton_roi_pool_kernel

@torch.library.register_kernel("torchvision::nms", "xpu")
def xpu_triton_roi_pool(
    rois,
    input,
    spatial_scale,
    output_size,
):
    num_rois = rois.size(0)
    num_channels = input.size(1)
    height = input.size(2)
    width = input.size(3)

    output_size = (num_rois, num_channels, output_size[0], output_size[1])
    output = torch.zeros(output_size, dtype=input.dtype, device=input.device)
    argmax = torch.zeros(output_size, dtype=torch.int, device=input.device)

    if (output.numel() == 0):
        return output, argmax

    input = input.contiguous()
    rois = rois.contiguous()

    nthreads = num_rois
    grid = lambda meta: (triton.cdiv(meta.nthreads, 256),)

    return triton_roi_pool_kernel[grid](
        input, rois, output, argmax,
        spatial_scale, num_channels, height, width, output_size[0], output_size[1],
        nthreads, BLOCK_SIZE=64
    )