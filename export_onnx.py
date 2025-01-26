import os.path
import sys
import torch
import onnx
import argparse
import onnxsim
from src.Module import ImageEncoder
from src.Module import MemAttention
from src.Module import MemEncoder
from src.Module import ImageDecoder
from sam2.build_sam import build_sam2
from sam2.build_sam import build_sam2_video_predictor
print(f"Python version: {sys.version}, {sys.version_info} ")
print(f"Pytorch version: {torch.__version__} ")
print(f"Onnx version: {onnx.__version__}")

def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

def export_image_encoder(model, onnx_path, IsSimplify=False):
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
    # 简化模型
    if IsSimplify:
        try:
            print('img_encoder {} simplifying with onnx-simplifier {}'.format(colorstr('ONNX:'), onnxsim.__version__))
            original_model = onnx.load(onnx_path+"image_encoder_2.onnx")
            simplified_model, check = onnxsim.simplify(original_model)
            assert check, "Simplified ONNX model could not be validated..."
            onnx.save(simplified_model, onnx_path+"image_encoder_2.onnx")
            print('{} simplify export success, saved as {} '.format(colorstr('ONNX:'), onnx_path+"image_encoder_2.onnx"))
        except Exception as e:
            print('img_encoder {} simplifier failure {}'.format(colorstr('ONNX:'), e))

    # 检查检查.onnx格式是否正确
    onnx_model = onnx.load(onnx_path+"image_encoder_2.onnx")
    onnx.checker.check_model(onnx_model)
    print(onnx.helper.printable_graph(onnx_model.graph))
    print("image_encoder_2.onnx model is valid!")
    
def export_memory_attention(model, onnx_path, Isdynamic=True, IsSimplify=True):
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
        onnx_path + "memory_attention_dynamic_2.onnx" if Isdynamic is True else onnx_path + "memory_attention_2.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names= input_name,
        output_names=["mem_atten_pix_feat_with_mem"],
        dynamic_axes = dynamic_axes if Isdynamic is True else None
    )
    # 简化模型,
    if IsSimplify:
        try:
            print('memory_attention {} simplifying with onnx-simplifier {}'.format(colorstr('ONNX:'), onnxsim.__version__))
            original_model = onnx.load(onnx_path+"memory_attention_dynamic_2.onnx" if Isdynamic is True else onnx_path + "memory_attention_2.onnx")
            simplified_model, check = onnxsim.simplify(original_model)
            onnx.save(simplified_model, onnx_path+"memory_attention_dynamic_2.onnx" if Isdynamic is True else onnx_path + "memory_attention_2.onnx")
            print('{} simplify export success, saved as {} '.format(colorstr('ONNX:'), "memory_attention_dynamic_2.onnx" if Isdynamic is True else "memory_attention_2.onnx"))
        except Exception as e:
            print('memory_attention {} simplifier failure {}'.format(colorstr('ONNX:'), e))

    # 检查检查.onnx格式是否正确
    onnx_model = onnx.load(onnx_path+"memory_attention_dynamic_2.onnx" if Isdynamic is True else onnx_path + "memory_attention_2.onnx")
    onnx.checker.check_model(onnx_model)
    print(onnx.helper.printable_graph(onnx_model.graph))
    print("memory_attention_dynamic_2.onnx or memory_attention_2.onnx model is valid!")

def export_image_decoder(model, onnx_path, Isdynamic=True, IsSimplify=True):
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
        onnx_path+"image_decoder_dynamic_2.onnx" if Isdynamic is True else onnx_path+"image_decoder_2.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names= input_name,
        output_names=output_name,
        dynamic_axes = dynamic_axes if Isdynamic is True else None
    )
    # 简化模型,
    if IsSimplify:
        try:
            print('image_decoder {} simplifying with onnx-simplifier {}'.format(colorstr('ONNX:'), onnxsim.__version__))
            original_model = onnx.load(onnx_path+"image_decoder_dynamic_2.onnx" if Isdynamic is True else onnx_path+"image_decoder_2.onnx")
            simplified_model, check = onnxsim.simplify(original_model)
            onnx.save(simplified_model, onnx_path+"image_decoder_dynamic_2.onnx" if Isdynamic is True else onnx_path+"image_decoder_2.onnx")
            print('{} simplify export success, saved as {} '.format(colorstr('ONNX:'), "image_decoder_dynamic_2.onnx" if Isdynamic is True else "image_decoder_2.onnx"))
        except Exception as e:
            print('image_decoder {} simplifier failure {}'.format(colorstr('ONNX:'), e))

    # 检查检查.onnx格式是否正确
    onnx_model = onnx.load(onnx_path+"image_decoder_dynamic_2.onnx" if Isdynamic is True else onnx_path+"image_decoder_2.onnx")
    onnx.checker.check_model(onnx_model)
    print(onnx.helper.printable_graph(onnx_model.graph))
    print("image_decoder_dynamic_2.onnx or image_decoder_2.onnx model is valid!")

def export_memory_encoder(model, onnx_path, IsSimplify=True):
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
    # 简化模型
    if IsSimplify:
        try:
            print('memory_encoder {} simplifying with onnx-simplifier {}'.format(colorstr('ONNX:'), onnxsim.__version__))
            original_model = onnx.load(onnx_path + "memory_encoder_2.onnx")
            simplified_model, check = onnxsim.simplify(original_model)
            onnx.save(simplified_model, onnx_path + "memory_encoder_2.onnx")
            print('{} simplify export success, saved as {} '.format(colorstr('ONNX:'), onnx_path + "memory_encoder_2.onnx"))
        except Exception as e:
            print('memory_encoder {} simplifier failure {}'.format(colorstr('ONNX:'), e))

    # 检查检查.onnx格式是否正确
    onnx_model = onnx.load(onnx_path+"memory_encoder_2.onnx")
    onnx.checker.check_model(onnx_model)
    print(onnx.helper.printable_graph(onnx_model.graph))
    print("memory_encoder_2.onnx model is valid!")

#****************************************************************************
model_type = ["tiny","small","large","base+"][0]
#onnx_output_path = "checkpoints/{}_simplify/".format(model_type)
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

    ### 这个不能简化，不然使用TRT加速时结果不对
    image_encoder = ImageEncoder(sam2_model).cpu()
    export_image_encoder(image_encoder, args.outdir, IsSimplify=False)

    image_decoder = ImageDecoder(sam2_model).cpu()
    export_image_decoder(image_decoder, args.outdir, Isdynamic=True, IsSimplify=True)

    mem_attention = MemAttention(sam2_model).cpu()
    export_memory_attention(mem_attention, args.outdir, Isdynamic=True, IsSimplify=True)

    mem_encoder   = MemEncoder(sam2_model).cpu()
    export_memory_encoder(mem_encoder, args.outdir, IsSimplify=True)