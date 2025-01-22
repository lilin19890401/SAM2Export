import torch
from torch import nn
from typing import Any
from sam2.build_sam import build_sam2
from sam2.modeling.sam2_base import SAM2Base
from sam2.utils.misc import fill_holes_in_mask_scores

class ImageEncoder(nn.Module):
    def __init__(self, sam_model: SAM2Base) -> None:
        super().__init__()
        self.model = sam_model
        self.no_mem_embed = sam_model.no_mem_embed #[1,1,256]
        self.image_encoder = sam_model.image_encoder
        self.num_feature_levels = sam_model.num_feature_levels
        self.prepare_backbone_features = sam_model. _prepare_backbone_features
        self.use_high_res_features_in_sam = sam_model.use_high_res_features_in_sam

    @torch.no_grad()
    def forward(self, image: torch.Tensor) ->tuple[torch.Tensor, torch.Tensor, torch.Tensor,torch.Tensor,torch.Tensor]:
        """Get the image feature on the input image."""
        backbone_out = self.image_encoder(image) # {"vision_features","vision_pos_enc","backbone_fpn"}
        if self.use_high_res_features_in_sam:
            # precompute projected level 0 and level 1 features in SAM decoder to avoid running it again on every SAM click
            backbone_out["backbone_fpn"][0] = self.model.sam_mask_decoder.conv_s0(backbone_out["backbone_fpn"][0])
            backbone_out["backbone_fpn"][1] = self.model.sam_mask_decoder.conv_s1(backbone_out["backbone_fpn"][1])

        vision_pos_enc = backbone_out["vision_pos_enc"] # 有3个tensor [1,256,256,256] [1,256,128,128] [1,256,64,64]
        backbone_fpn = backbone_out["backbone_fpn"]     # 有3个tensor [1,32,256,256] [1,64,128,128] [1,256,64,64]
        pix_feat = backbone_out["vision_features"]      # 有1个tensor [1,256,64,64]

        # expand the features to have the same dimension as the number of objects
        expanded_backbone_out = {
            "backbone_fpn": backbone_fpn,
            "vision_pos_enc": vision_pos_enc,
        }
        for i, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
            expanded_backbone_out["backbone_fpn"][i] = feat.expand(1, -1, -1, -1)
        for i, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
            expanded_backbone_out["vision_pos_enc"][i] = pos.expand(1, -1, -1, -1)

        """Prepare and flatten visual features."""
        (_, current_vision_feats, current_vision_pos_embeds, feat_sizes) = self.prepare_backbone_features(expanded_backbone_out)
        # current_vision_feats[3]: [65536,1,32] [16384,1,64] [4096,1,256]
        # current_vision_pos_embeds[3]: [65536,1,32] [16384,1,64] [4096,1,256]
        # feat_sizes[3]: [(256,256), (128,128), (64,64)]

        # directly add no-mem embedding (instead of using the transformer encoder)
        current_vision_feat = current_vision_feats[-1] + self.no_mem_embed
        #current_vision_feat2 = current_vision_feat.reshape(64,64,1,256).permute(2, 3, 0, 1) # [1,256,64,64]
        current_vision_feat2 = current_vision_feat.reshape(feat_sizes[2][0], feat_sizes[2][1], current_vision_feat.size(1), current_vision_feat.size(2)).permute(2, 3, 0, 1)  # [1,256,64,64]

        # High-resolution feature maps for the SAM head, reshape (HW)BC => BCHW
        #high_res_features_0 = current_vision_feats[0].reshape(256,256, 1, 32).permute(2, 3, 0, 1) # [1, 32, 256, 256]
        #high_res_features_1 = current_vision_feats[1].reshape(128,128, 1, 64).permute(2, 3, 0, 1) # [1, 64, 128, 128]
        high_res_features_0 = current_vision_feats[0].reshape(feat_sizes[0][0], feat_sizes[0][1], current_vision_feats[0].size(1), current_vision_feats[0].size(2)).permute(2, 3, 0, 1) # [1, 32, 256, 256]
        high_res_features_1 =current_vision_feats[1].reshape(feat_sizes[1][0], feat_sizes[1][1],current_vision_feats[1].size(1), current_vision_feats[1].size(2)).permute(2, 3, 0, 1) # [1, 64, 128, 128]

        # pix_feat                      [1, 256, 64, 64]    图像层面提取的像素特征
        # high_res_features_0           [1, 32, 256, 256]   level 0层面的高分辨率特征
        # high_res_features_1           [1, 64, 128, 128]   level 1层面的高分辨率特征
        # current_vision_feat2          [1, 256, 64, 64]    for initial conditioning frames, encode them without using any previous memory；for conditioning frames, we should meege previous memory
        # current_vision_pos_embeds[-1] [4096, 1, 256]
        return pix_feat, high_res_features_0, high_res_features_1, current_vision_feat2, current_vision_pos_embeds[-1]

class MemAttention(nn.Module):
    def __init__(self, sam_model: SAM2Base) -> None:
        super().__init__()
        self.model = sam_model
        self.no_mem_embed = sam_model.no_mem_embed
        self.memory_attention = sam_model.memory_attention

    # @torch.no_grad()
    def forward(
        self,
        current_vision_feat: torch.Tensor,          # [1, 256, 64, 64], 当前帧的视觉特征
        current_vision_pos_embed: torch.Tensor,     # [4096, 1, 256], 当前帧的位置特征
        memory_0:torch.Tensor,                      # [num_obj_ptr=16,256]->[num_obj_ptr=16,4,64]->[4*num_obj_ptr,1,64], 第一帧的obj_prt[1,256]和最近的15帧的obj_ptr[15,256]进行cat（第一行固定是1st的obj_prt,之后行是按照帧距离排列）
        memory_1:torch.Tensor,                      # [n=7,64,64,64]->[n=7,64,4096]->[4096n,1,64],  之前缓存的前7帧
        memory_pos_embed:torch.Tensor               # [n*4096,1,64], 最近n=7帧的位置编码特性
    ) -> tuple[Any]:
        num_obj_ptr_tokens =  memory_0.shape[0]*4
        current_vision_feat=current_vision_feat.permute(2,3,0,1).reshape(4096,1,256)
        current_vision_feat = current_vision_feat - self.no_mem_embed

        memory_0 = memory_0.reshape(-1,1,4,64)                  #[16,1,4,64]
        memory_0 = memory_0.permute(0, 2, 1, 3).flatten(0, 1)   #[64,1,64]

        memory_1 = memory_1.view(-1, 64, 64*64).permute(0,2,1)  #[7,4096,64]
        memory_1 = memory_1.reshape(-1,1,64)                    #[28672,1,64]

        print(memory_0.shape,memory_1.shape)
        memory = torch.cat((memory_1,memory_0),dim=0)
        pix_feat_with_mem = self.memory_attention(
            curr = current_vision_feat,
            curr_pos = current_vision_pos_embed,
            memory = memory,
            memory_pos = memory_pos_embed,
            num_obj_ptr_tokens= num_obj_ptr_tokens,
        )
        # reshape the output (HW)xBxC => BxCxHxW
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(1, 256, 64, 64) # [1,256,64,64]
        return pix_feat_with_mem #[1,256,64,64]

class MemEncoder(nn.Module):
    def __init__(self, sam_model: SAM2Base) -> None:
        super().__init__()
        self.model = sam_model
        self.maskmem_tpos_enc = sam_model.maskmem_tpos_enc
        self.feat_sizes = [(256, 256), (128, 128), (64, 64)]
    @torch.no_grad()
    def forward(
        self,
        mask_for_mem: torch.Tensor,  # [1,1,1024,1024]
        pix_feat: torch.Tensor,      # [1,256,64,64]
        mask_from_pts: torch.Tensor, # [1,1]
    )-> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        maskmem_features, maskmem_pos_enc = self.model._encode_new_memory(
            current_vision_feats=pix_feat,
            feat_sizes=self.feat_sizes,
            pred_masks_high_res=mask_for_mem,
            is_mask_from_pts=mask_from_pts == torch.tensor([[1]]),
        )
        print(maskmem_features.shape)   # [1,64,64,64]
        # maskmem_features = maskmem_features.view(1, 64, 64*64).permute(2, 0, 1)
        maskmem_pos_enc = maskmem_pos_enc.view(1, 64, 64*64).permute(2, 0, 1)

        return maskmem_features, maskmem_pos_enc, self.maskmem_tpos_enc, mask_from_pts

class ImageDecoder(nn.Module):
    def __init__(self, sam_model: SAM2Base) -> None:
        super().__init__()
        self.model = sam_model
        self.sigmoid_scale_for_mem_enc = sam_model.sigmoid_scale_for_mem_enc
        self.sigmoid_bias_for_mem_enc = sam_model.sigmoid_bias_for_mem_enc
    @torch.no_grad()
    def forward(
        self,
        point_coords: torch.Tensor,         # [num_labels,num_points,2]         handle point prompts
        point_labels: torch.Tensor,         # [num_labels,num_points]
        pix_feat_with_mem: torch.Tensor,    # [1,256,64,64]         fused the visual feature with previous memory features in the memory bank
        high_res_feats_0: torch.Tensor,     # [1, 32, 256, 256]     High-resolution feature
        high_res_feats_1: torch.Tensor,     # [1, 64, 128, 128]
    ):
        point_inputs = {"point_coords":point_coords,"point_labels":point_labels}
        high_res_feats = [high_res_feats_0, high_res_feats_1]

        sam_outputs = self.model._forward_sam_heads(
            backbone_features=pix_feat_with_mem,
            point_inputs=point_inputs,
            mask_inputs=None,
            high_res_features=high_res_feats,
            multimask_output=True
        )
        (
            _,                      # low_res_multimasks [1,3,256,256]
            _,                      # high_res_multimasks [1,3,1024,1024]
            _,                      # ious [1,3]
            low_res_masks,          # [1,1,256,256]
            high_res_masks,         # [1,1,1024,1024]
            obj_ptr,                # [1,256]
            object_score_logits,    # [1,1]
        ) = sam_outputs
        # 处理高分辨率mask
        # apply sigmoid on the raw mask logits to turn them into range (0, 1).这里的值给后续的MemoryEncoder使用
        mask_for_mem = torch.sigmoid(high_res_masks)
        mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc
        # potentially fill holes in the predicted masks
        pred_mask = fill_holes_in_mask_scores(low_res_masks, 8)
        # 还原到模型输入大小
        # pred_mask = torch.nn.functional.interpolate(
        #     pred_mask,
        #     size=(high_res_masks.shape[2], high_res_masks.shape[3]),
        #     mode="bilinear",
        #     align_corners=False,
        # )

        # obj_ptr                       [1, 256]
        # mask_for_mem                  [1, 1, 1024, 1024]
        # pred_mask                     [1, 1, 1024, 1024]
        return obj_ptr, mask_for_mem, pred_mask,