import os
from PIL import Image
import numpy as np
import torch
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import matplotlib.pyplot as plt

def show_mask(mask, ax, obj_id=None, random_color=False, borders=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        cv2.drawContours(mask_image, contours, -1, (0, 0, 1, 0.4), thickness=1) 
    ax.imshow(mask_image)

# Setup
video_path = "notebooks/videos/bedroom"
save_path = "notebooks/videos/bedroom_segmented"
sam2_checkpoint = "checkpoints/sam2_hiera_tiny.pt"
model_cfg = "sam2_hiera_t.yaml"
device = "cuda"

# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_path
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# Auto-masking of first frame (from automatic mask generation notebook)
sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
first_frame_path = os.path.join(video_path, os.listdir(video_path)[0])
first_frame = Image.open(first_frame_path)
first_frame = np.array(first_frame.convert("RGB"))
mask_generator = SAM2AutomaticMaskGenerator(sam2)
auto_masks = mask_generator.generate(first_frame)
print("Number of auto-masks:", len(auto_masks))

# Add every 'auto-mask' as it's own prompt for video tracking
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
inference_state = predictor.init_state(video_path=video_path)
dtype = next(predictor.parameters()).dtype
lowres_side_length = predictor.image_size // 4
for mask_idx, mask_result in enumerate(auto_masks):

    # Get mask into form expected by the model
    mask_tensor = torch.tensor(mask_result["segmentation"], dtype=dtype, device=device)
    lowres_mask = torch.nn.functional.interpolate(
        mask_tensor.unsqueeze(0).unsqueeze(0),
        size=(lowres_side_length, lowres_side_length),
        mode="bilinear",
        align_corners=False,
    ).squeeze()

    # Add each mask as it's own 'object' to segment
    _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=mask_idx,
        mask=lowres_mask,
    )

# Do video segmentation (same as video segmentation notebook)
video_segments = {}
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)
    }

# Save video segments
os.makedirs(save_path, exist_ok=True)
for out_frame_idx, frame_name in enumerate(frame_names):
    frame_path = os.path.join(video_path, frame_name)
    frame = Image.open(frame_path)
    frame = np.array(frame.convert("RGB"))
    frame_segments = video_segments.get(out_frame_idx, {})
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(frame)
    for obj_id, mask in frame_segments.items():
        show_mask(mask, ax, obj_id=obj_id, random_color=True, borders=True)
    plt.axis("off")
    plt.savefig(os.path.join(save_path, frame_name))
    plt.close(fig)