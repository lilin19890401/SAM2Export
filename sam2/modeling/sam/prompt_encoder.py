# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, Type

import torch
from torch import nn

from sam2.modeling.position_encoding import PositionEmbeddingRandom

from sam2.modeling.sam2_utils import LayerNorm2d


class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder. 对prompts(点、框、mask)进行embeding

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding, as (H, W).
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
          mask_in_chans (int): The number of hidden channels used for
            encoding input masks.
          activation (nn.Module): The activation to use when encoding
            input masks.
        """
        super().__init__()
        self.embed_dim = embed_dim
        # 用于坐标归一化
        self.input_image_size = input_image_size
        # 用于图像经过patch处理之后的width height，容易和embed_dim混淆
        self.image_embedding_size = image_embedding_size
        # PE, sin-cos
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        # 统一point和box的embeding向量
        self.num_point_embeddings: int = 4  # pos/neg point + 2 box corners
        # 对point是否有效进行embeding,然后加到PE上
        point_embeddings = [ nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings) ]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        #如果point为无效的（比如pad的），则用下面这个
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        self.mask_input_size = (
            4 * image_embedding_size[0],
            4 * image_embedding_size[1],
        )
        # mask下采样模块
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        pad: bool,
    ) -> torch.Tensor:
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        # pad,保持和box统一
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(
            points, self.input_image_size
        )

        # onnx broadcast error
        #point_embedding[labels == -1] = 0.0
        #point_embedding[labels == -1] += self.not_a_point_embed.weight

        # 这个适用于onnx,不适用于tfile
        #point_embedding[labels == -1] = self.not_a_point_embed.weight
        
        #point_embedding[labels == 0] += self.point_embeddings[0].weight
        #point_embedding[labels == 1] += self.point_embeddings[1].weight
        #point_embedding[labels == 2] += self.point_embeddings[2].weight
        #point_embedding[labels == 3] += self.point_embeddings[3].weight

        # 这个也适用于tfile
        #labels = labels.int()
        # table = torch.zeros((5, self.point_embeddings[0].weight.shape[1]))
        # table[0] = self.not_a_point_embed.weight
        # table[1] = self.point_embeddings[0].weight
        # table[2] = self.point_embeddings[1].weight
        # table[3] = self.point_embeddings[2].weight
        # table[4] = self.point_embeddings[3].weight
        # for i in range(labels.shape[0]):
        #     point_embedding[i] = point_embedding[i] + table[labels[i] + 1]

        labels = labels.int()
        mask_neg1 = (labels == -1).unsqueeze(-1).expand_as(point_embedding)
        mask_0 = (labels == 0).unsqueeze(-1).expand_as(point_embedding)
        mask_1 = (labels == 1).unsqueeze(-1).expand_as(point_embedding)
        mask_2 = (labels == 2).unsqueeze(-1).expand_as(point_embedding)
        mask_3 = (labels == 3).unsqueeze(-1).expand_as(point_embedding)

        # Apply the weights according to the mask
        point_embedding = torch.where(mask_neg1, self.not_a_point_embed.weight.expand_as(point_embedding),point_embedding)
        point_embedding = torch.where(mask_0,point_embedding + self.point_embeddings[0].weight.expand_as(point_embedding),point_embedding)
        point_embedding = torch.where(mask_1,point_embedding + self.point_embeddings[1].weight.expand_as(point_embedding),point_embedding)
        point_embedding = torch.where(mask_2,point_embedding + self.point_embeddings[2].weight.expand_as(point_embedding),point_embedding)
        point_embedding = torch.where(mask_3,point_embedding + self.point_embeddings[3].weight.expand_as(point_embedding),point_embedding)

        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Embeds box prompts."""
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(
            coords, self.input_image_size
        )
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Embeds mask inputs."""
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    def _get_batch_size(
        self,
        coords: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if coords is not None and labels is not None:
            return coords.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    def forward(
        self,
        coords: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
        masks_enable: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
            and labels to embed.
          boxes (torch.Tensor or none): boxes to embed
          masks (torch.Tensor or none): masks to embed

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
          torch.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)
        """
        if coords is None or labels is None:
            raise("onnx not supported coords is None")

        bs = self._get_batch_size(coords, labels, masks)
        # sparse_embeddings只是为了保证函数返回形式统一，即点和框都为NONE的时候返回一个空的tensor
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self._get_device())     # [B,4,256]

        point_embeddings = self._embed_points(coords, labels, pad=True)
        sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        
        dense_embeddings1 = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
        )
        dense_embeddings2 = self._embed_masks(masks)

        dense_embeddings = torch.where(masks_enable[0] == 1, dense_embeddings2, dense_embeddings1)  # [B,256,64,64]

        return sparse_embeddings, dense_embeddings, self.get_dense_pe()
