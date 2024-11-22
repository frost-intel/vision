import torch
import torchvision.ops
import triton
import triton.language as tl

@triton.jit
def triton_roi_pool_kernel(
    input_ptr, rois_ptr, output_ptr, argmax_data_ptr,
    spatial_scale, channels, height, width, pooled_height, pooled_width, nthreads,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    index = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = index < nthreads

    pw = index % pooled_width
    ph = (index // pooled_width) % pooled_height
    c = (index // pooled_width // pooled_height) % channels
    n = index // pooled_width // pooled_height // channels

    offset_rois = tl.load(rois_ptr + n * 5, mask=mask)
    roi_batch_ind = offset_rois[0]
    roi_start_w = tl.extra.intel.libdevice.round(offset_rois[1] * spatial_scale)
    roi_start_h = tl.extra.intel.libdevice.round(offset_rois[2] * spatial_scale)
    roi_end_w = tl.extra.intel.libdevice.round(offset_rois[3] * spatial_scale)
    roi_end_h = tl.extra.intel.libdevice.round(offset_rois[4] * spatial_scale)

    roi_width = tl.max(roi_end_w - roi_start_w + 1, 1)
    roi_height = tl.max(roi_end_h - roi_start_h + 1, 1)
    bin_size_h = roi_height / pooled_height
    bin_size_w = roi_width / pooled_width

    hstart = tl.math.floor(ph * bin_size_h).to(tl.int32)
    wstart = tl.math.floor(pw * bin_size_w).to(tl.int32)
    hend = tl.math.ceil((ph + 1) * bin_size_h).to(tl.int32)
    wend = tl.math.ceil((pw + 1) * bin_size_w).to(tl.int32)

    hstart = tl.min(tl.max(hstart + roi_start_h, 0), height)
    hend = tl.min(tl.max(hend + roi_start_h, 0), height)
    wstart = tl.min(tl.max(wstart + roi_start_w, 0), width)
    wend = tl.min(tl.max(wend + roi_start_w, 0), width)
    is_empty = (hend <= hstart) | (wend <= wstart)

    maxval = tl.where(is_empty, 0.0, -float('inf'))
    maxidx = tl.where(is_empty, -1, 0)

    offset_input = input_ptr + (roi_batch_ind * channels + c) * height * width
    for h in range(hstart, hend):
        for w in range(wstart, wend):
            input_index = h * width + w
            val = tl.load(offset_input + input_index, mask=mask)
            maxval = tl.where(val > maxval, val, maxval)
            maxidx = tl.where(val > maxval, input_index, maxidx)

    tl.store(output_ptr + index, maxval, mask=mask)
    tl.store(argmax_data_ptr + index, maxidx, mask=mask)