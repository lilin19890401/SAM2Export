import os.path

import torch
import onnx
import argparse
from onnxsim import simplify
from src.Module import ImageEncoder
from src.Module import MemAttention
from src.Module import MemEncoder
from src.Module import ImageDecoder
from sam2.build_sam import build_sam2
from sam2.build_sam import build_sam2_video_predictor

def export_image_encoder(model,onnx_path):
    input_img = torch.randn(1, 3,1024, 1024).cpu()
    out = model(input_img)
    output_names = ["img_encoder_pix_feat", "img_encoder_high_res_feat0", "img_encoder_high_res_feat1", "img_encoder_vision_feats", "img_encoder_vision_pos_embed"]
    torch.onnx.export(
        model,
        input_img,
        onnx_path+"image_encoder_2.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["img_encoder_image"],
        output_names=output_names,
    )
    # # 简化模型, tmd将我的输出数量都简化掉一个，sb
    # original_model = onnx.load(onnx_path+"image_encoder.onnx")
    # simplified_model, check = simplify(original_model)
    # onnx.save(simplified_model, onnx_path+"image_encoder.onnx")
    # 检查检查.onnx格式是否正确
    onnx_model = onnx.load(onnx_path+"image_encoder_2.onnx")
    onnx.checker.check_model(onnx_model)
    print("image_encoder_2.onnx model is valid!")
    
def export_memory_attention(model,onnx_path):
    current_vision_feat = torch.randn(1,256,64,64)      #[1, 256, 64, 64],当前帧的视觉特征
    current_vision_pos_embed = torch.randn(4096,1,256)  #[4096, 1, 256],当前帧的位置特征
    memory_0 = torch.randn(16,256)                   
    memory_1 = torch.randn(7,64,64,64)
    memory_pos_embed = torch.randn(28736,1,64)      #[n*4096+64,1,64], 最近n=7帧的位置编码特性
    out = model(
            current_vision_feat = current_vision_feat,
            current_vision_pos_embed = current_vision_pos_embed,
            memory_0 = memory_0,
            memory_1 = memory_1,
            memory_pos_embed = memory_pos_embed
        )
    input_name = ["mem_atten_current_vision_feat",
                "mem_atten_current_vision_pos_embed",
                "mem_atten_memory_0",
                "mem_atten_memory_1",
                "mem_atten_memory_pos_embed"]
    dynamic_axes = {
        "mem_atten_memory_0": {0: "num"},
        "mem_atten_memory_1": {0: "buff_size"},
        "mem_atten_memory_pos_embed": {0: "pos_embed_buff_size"},
    }
    torch.onnx.export(
        model,
        (current_vision_feat,current_vision_pos_embed,memory_0,memory_1,memory_pos_embed),
        onnx_path+"memory_attention_dynamic_2.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names= input_name,
        output_names=["mem_atten_pix_feat_with_mem"],
        dynamic_axes = dynamic_axes
    )
     # 简化模型,
    # original_model = onnx.load(onnx_path+"memory_attention.onnx")
    # simplified_model, check = simplify(original_model)
    # onnx.save(simplified_model, onnx_path+"memory_attention.onnx")
    # 检查检查.onnx格式是否正确
    onnx_model = onnx.load(onnx_path+"memory_attention_dynamic_2.onnx")
    onnx.checker.check_model(onnx_model)
    print("memory_attention_dynamic_2.onnx model is valid!")

def export_image_decoder(model,onnx_path):
    point_coords = torch.randn(1,1,2).cpu()         # point_coords = torch.randn(1,2,2).cpu()
    point_labels = torch.randn(1,1).cpu()           # point_labels = torch.randn(1,2).cpu()
    pix_feat_with_mem = torch.randn(1,256,64,64).cpu()
    high_res_feats_0 = torch.randn(1,32,256,256).cpu()
    high_res_feats_1 = torch.randn(1,64,128,128).cpu()

    out = model(
        point_coords = point_coords,
        point_labels = point_labels,
        pix_feat_with_mem = pix_feat_with_mem,
        high_res_feats_0 = high_res_feats_0,
        high_res_feats_1 = high_res_feats_1
    )
    input_name = ["img_decoder_point_coords","img_decoder_point_labels","img_decoder_pix_feat_with_mem","img_decoder_high_res_feats_0","img_decoder_high_res_feats_1"]
    output_name = ["img_decoder_obj_ptr","img_decoder_mask_for_mem","img_decoder_pred_mask"]
    dynamic_axes = {
        "img_decoder_point_coords":{1:"num_points"},
        "img_decoder_point_labels": {1:"num_points"}
    }
    ###  point_coords和point_labels可以选择使用动态或者非动态写死的方式
    torch.onnx.export(
        model,
        (point_coords,point_labels,pix_feat_with_mem,high_res_feats_0,high_res_feats_1),
        onnx_path+"image_decoder_dynamic_2.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names= input_name,
        output_names=output_name,
        dynamic_axes = dynamic_axes
    )
    # 简化模型,
    # original_model = onnx.load(onnx_path+"image_decoder.onnx")
    # simplified_model, check = simplify(original_model)
    # onnx.save(simplified_model, onnx_path+"image_decoder.onnx")
    # 检查检查.onnx格式是否正确
    onnx_model = onnx.load(onnx_path+"image_decoder_dynamic_2.onnx")
    onnx.checker.check_model(onnx_model)
    print("image_decoder_dynamic_2.onnx model is valid!")

def export_memory_encoder(model,onnx_path):
    mask_for_mem = torch.randn(1,1,1024,1024) 
    pix_feat = torch.randn(1,256,64,64)
    mask_from_pts = torch.randn(1,1)

    out = model(mask_for_mem = mask_for_mem,pix_feat = pix_feat, mask_from_pts=mask_from_pts)

    input_names = ["mem_encoder_mask_for_mem","mem_encoder_pix_feat","mem_encoder_mask_from_pts" ]
    output_names = ["mem_encoder_maskmem_features","mem_encoder_maskmem_pos_enc","mem_encoder_temporal_code","mem_encoder_is_mask_from_pts"]
    torch.onnx.export(
        model,
        (mask_for_mem,pix_feat,mask_from_pts),
        onnx_path+"memory_encoder_2.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names= input_names,
        output_names= output_names
    )
    # 简化模型,
    # original_model = onnx.load(onnx_path+"memory_encoder.onnx")
    # simplified_model, check = simplify(original_model)
    # onnx.save(simplified_model, onnx_path+"memory_encoder.onnx")
    # 检查检查.onnx格式是否正确
    onnx_model = onnx.load(onnx_path+"memory_encoder_2.onnx")
    onnx.checker.check_model(onnx_model)
    print("memory_encoder_2.onnx model is valid!")

#****************************************************************************
model_type = ["tiny","small","large","base+"][1]
onnx_output_path = "checkpoints/{}/".format(model_type)
model_config_file = "sam2_hiera_{}.yaml".format(model_type)
model_checkpoints_file = "checkpoints/sam2_hiera_{}.pt".format(model_type)
if not os.path.exists(onnx_output_path):
    os.makedirs(onnx_output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="导出SAM2为onnx文件")
    parser.add_argument("--outdir",type=str,default=onnx_output_path,required=False,help="path")
    parser.add_argument("--config",type=str,default=model_config_file,required=False,help="*.yaml")
    parser.add_argument("--checkpoint",type=str,default=model_checkpoints_file,required=False,help="*.pt")
    args = parser.parse_args()
    # sam2_model = build_sam2(args.config, args.checkpoint, device="cpu")
    sam2_model = build_sam2_video_predictor(args.config, args.checkpoint, device="cpu")
    
    image_encoder = ImageEncoder(sam2_model).cpu()
    export_image_encoder(image_encoder,args.outdir)

    image_decoder = ImageDecoder(sam2_model).cpu()
    export_image_decoder(image_decoder,args.outdir)


    mem_attention = MemAttention(sam2_model).cpu()
    export_memory_attention(mem_attention,args.outdir)

    mem_encoder   = MemEncoder(sam2_model).cpu()
    export_memory_encoder(mem_encoder,args.outdir)