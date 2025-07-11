import math
from typing import Optional, List, Tuple, Sequence
import einops
import mmengine
import numpy as np
from peft import LoraConfig, get_peft_model
from torchvision.transforms.functional import normalize
import torch
from torch import Tensor, nn
from transformers.models.sam.modeling_sam import SamPositionalEmbedding
from mmengine.model import BaseModule
from transformers import AutoModel, SamModel, AutoProcessor
from mmseg.models import EncoderDecoder, accuracy
from mmseg.models.utils import resize
from mmseg.registry import MODELS
from mmseg.structures import build_pixel_sampler
from mmseg.utils import ConfigType, OptConfigType, OptMultiConfig, SampleList, MultiConfig
import torch.nn.functional as F
from refseg.models.losses import MILCrossEntropy


@MODELS.register_module()
class RefSegEncoderDecoder(EncoderDecoder):
    def __init__(
            self,
            lora_cfg: dict,
            clip_vision_encoder: dict,
            clip_text_encoder: dict,
            prompter: dict=None,
            *args,
            **kwargs):
        super().__init__(*args, **kwargs)
        self.lora_cfg = lora_cfg

        self.clip_vision_encoder = MODELS.build(clip_vision_encoder)
        self.clip_text_encoder = MODELS.build(clip_text_encoder)
        self.prompter = MODELS.build(prompter)

        if self.backbone.is_sam2:
            self.lora_cfg.pop('backbone.model.vision_encoder', None)
        else:
            self.lora_cfg.pop('backbone.model.image_encoder', None)
        # set efficient finetuning parameters
        self.set_finetune_parameters()

        if hasattr(self.clip_text_encoder, 'projector'):
            self.clip_text_encoder.projector.requires_grad_(True)
        self.clip_text_encoder.logit_scale.requires_grad_(True)
        self.clip_text_encoder.logit_bias.requires_grad_(True)
        self.print_trainable_parameters()

    def set_finetune_parameters(self):
        # SAM Model: backbone
        # CLIP vision encoder: clip_vision_encoder
        # CLIP text encoder: clip_text_encoder
        peft_keys = [
            'backbone.model.vision_encoder',
            'backbone.model.image_encoder', 'clip_vision_encoder', 'clip_text_encoder'
        ]
        for k in peft_keys:
            if k in self.lora_cfg:
                v_ = self.lora_cfg[k].copy()
                lora_config = LoraConfig(**v_)

                parts = k.split('.')
                parent_parts = parts[:-1]
                attr_name = parts[-1]
                # 遍历到父对象
                current = self
                for part in parent_parts:
                    current = getattr(current, part)
                sub_module = getattr(current, attr_name)
                setattr(current, attr_name, get_peft_model(sub_module, lora_config))
            else:
                print(f"Warning: {k} not in lora_cfg")
        # import ipdb; ipdb.set_trace()
        print(f"Set peft keys: {peft_keys}, all finished")


    def get_nb_trainable_parameters(self) -> tuple[int, int]:
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            # Due to the design of 4bit linear layers from bitsandbytes
            # one needs to multiply the number of parameters by 2 to get
            # the correct number of parameters
            if param.__class__.__name__ == "Params4bit":
                if hasattr(param, "element_size"):
                    num_bytes = param.element_size()
                elif not hasattr(param, "quant_storage"):
                    num_bytes = 1
                else:
                    num_bytes = param.quant_storage.itemsize
                num_params = num_params * 2 * num_bytes

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param


    def print_trainable_parameters(self) -> None:
        trainable_params, all_param = self.get_nb_trainable_parameters()
        print(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.4f}"
        )

    def extract_feat(self, inputs, text_list, img_path_list=None) -> List[Tensor]:
        # for CLIP vision encoder
        x_clip = normalize(inputs, mean=self.clip_vision_encoder.processor.image_mean, std=self.clip_vision_encoder.processor.image_std)
        x_clip = F.interpolate(x_clip, size=list(self.clip_vision_encoder.processor.size.values()), mode='bilinear', align_corners=False)
        clip_visual_feat = self.clip_vision_encoder(x_clip)
        clip_visual_feat_pooler = clip_visual_feat['pooler_output']  # BX1152
        clip_visual_feat = clip_visual_feat['last_hidden_state']  # BX729X1152
        text_dict = self.clip_text_encoder.processor(text_list, return_tensors='pt', padding=True, truncation=True, max_length=128)
        text_dict = {k: v.to(inputs.device) for k, v in text_dict.items()}
        input_ids = text_dict['input_ids']  # BX14
        # attention_mask = text_dict['attention_mask']  # BX14
        clip_text_feat = self.clip_text_encoder(**text_dict)
        clip_text_feat_pooler = clip_text_feat['pooler_output']  # BX1152
        clip_text_feat = clip_text_feat['last_hidden_state']  # BX18X1152

        # print(f'clip_visual_feat: {clip_visual_feat.shape}')
        # print(f'clip_visual_feat_pooler: {clip_visual_feat_pooler.shape}')
        # print(f'clip_text_feat: {clip_text_feat.shape}')
        # print(f'clip_text_feat_pooler: {clip_text_feat_pooler.shape}')
        # for prompter
        results = self.prompter(
            clip_visual_feat=clip_visual_feat,
            clip_text_feat=clip_text_feat,
            clip_text_feat_pooler=clip_text_feat_pooler,
            clip_text_encoder=self.clip_text_encoder,
            sam_model=self.backbone,
            output_vis_feat=False,
        ) # BX10X256

        # for visualization
        if img_path_list is not None:
            output_vis_feats = results['output_vis_feats']
            for idx, img_path in enumerate(img_path_list):
                save_dict = {
                    k: v[idx] for k, v in output_vis_feats.items()
                }
                torch.save(save_dict, img_path)

        results.update({
            'inputs': inputs,
        })
        # for SAM vision encoder
        seg_results = self.backbone(**results)
        return seg_results


    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        text_list = [data_sample.get('text', '') for data_sample in data_samples]
        x = self.extract_feat(inputs, text_list)

        losses = dict()
        loss_decode = self._decode_head_forward_train(x, data_samples)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, data_samples)
            losses.update(loss_aux)

        return losses

    def predict(self, inputs: Tensor, data_samples):
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]
        batch_img_metas_with_text = []

        for img_meta, data_sample in zip(batch_img_metas, data_samples):
            img_meta['text'] = data_sample.get('text', '')
            batch_img_metas_with_text.append(img_meta)

        seg_logits = self.inference(inputs, batch_img_metas_with_text)
        return self.postprocess_result(seg_logits, data_samples)


    def encode_decode(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        text_list = [img_meta['text'] for img_meta in batch_img_metas]

        # ## for visualization
        # vis_folder = 'work_dirs/visualizer/vis_data'
        # os.makedirs(vis_folder, exist_ok=True)
        # img_path_list = [vis_folder + '/' + img_meta['img_path'].split('/')[-1].split('.')[0] + '.pth' for img_meta in batch_img_metas]
        img_path_list = None

        x = self.extract_feat(inputs, text_list, img_path_list)
        seg_logits = self.decode_head.predict(x, batch_img_metas, self.test_cfg)
        return seg_logits



@MODELS.register_module()
class RefSegSiglipTextModel(BaseModule):
    def __init__(
            self,
            model_name_or_path: str='thomas/siglip-so400m-patch14-384',
            cache_dir=None,  # cache_dir
            init_cfg=None,
            projector_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.model_name_or_path = model_name_or_path
        self.processor = AutoProcessor.from_pretrained(model_name_or_path, cache_dir=cache_dir).tokenizer
        model = AutoModel.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        self.model = model.text_model
        self.config = self.model.config
        self.logit_scale = model.logit_scale
        self.logit_bias = model.logit_bias
        self.model.is_init = True
        self.logit_scale.is_init = True
        self.logit_bias.is_init = True

        self.projector_cfg = projector_cfg
        if projector_cfg is not None:
            self.projector = nn.Linear(**projector_cfg)
            self.projector.is_init = True
            # copy weight and bias from head to projector
            self.projector.weight.data = self.model.head.weight.data.clone()
            self.projector.bias.data = self.model.head.bias.data.clone()

    def init_weights(self):
        pass

    def forward(self, *args, **kwargs):
        results = self.model(*args, **kwargs)
        if self.projector_cfg is not None:
            results.last_hidden_state = self.projector(results.last_hidden_state)
        return results


@MODELS.register_module()
class RefSegSam(BaseModule):
    def __init__(
            self,
            model_name_or_path: str='KyanChen/sam2.1-hiera-tiny',
            with_dense_prompt=False,
            cache_dir=None,
            init_cfg=None,
            **kwargs
    ):
        super().__init__(init_cfg=init_cfg)
        if 'sam2' in model_name_or_path:
            from sam2.build_sam import build_sam2_hf
            sam_model = build_sam2_hf(model_name_or_path, cache_dir=cache_dir, **kwargs)
        else:
            sam_model = SamModel.from_pretrained(model_name_or_path, cache_dir=cache_dir, **kwargs)

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        '''
        all modules in SAM2: 'obj_ptr_proj', 'mask_downsample', 'sam_prompt_encoder', 'no_obj_embed_spatial', 'obj_ptr_tpos_proj', 'image_encoder', 'sam_mask_decoder', 'no_obj_ptr', 'maskmem_tpos_enc', 'memory_attention', 'no_mem_pos_enc', 'no_mem_embed', 'memory_encoder'
        Used modules: 'no_mem_embed', 'image_encoder', 'sam_mask_decoder', 'sam_prompt_encoder', 
'       '''

        '''
        all modules in SAM1: shared_image_embedding, vision_encoder, prompt_encoder, mask_decoder
        '''
        all_modules = set([name.split('.')[0] for name, _ in sam_model.named_parameters()])

        if 'sam2' in model_name_or_path:
            used_modules = ['no_mem_embed', 'image_encoder', 'sam_mask_decoder', 'sam_prompt_encoder']
        else:
            used_modules = ['shared_image_embedding', 'vision_encoder', 'prompt_encoder', 'mask_decoder']
        unused_modules = all_modules - set(used_modules)
        # remove modules that are not used
        for unused_module_name in unused_modules:
            delattr(sam_model, unused_module_name)

        self.model = sam_model
        self.model.is_init = True
        self.with_dense_prompt = with_dense_prompt
        self.is_sam2 = 'sam2' in model_name_or_path


    def init_weights(self):
        pass

    def forward(self, inputs, sparse_prompts, dense_prompts, *args, **kwargs):
        inputs = normalize(inputs, self.mean, self.std)

        batch_size = inputs.size(0)

        if self.is_sam2:
            # SAM2 Encoder
            backbone_out = self.model.forward_image(inputs) # ['vision_features', 'vision_pos_enc', 'backbone_fpn']
            _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)  # backbone_out, vision_feats, vision_pos_embeds, feat_sizes
            # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
            if self.model.directly_add_no_mem_embed:
                vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed
            feats = [einops.rearrange(x, '(h w) b c -> b c h w', h=int(math.sqrt(x.shape[0])), w=int(math.sqrt(x.shape[0]))) for x in vision_feats]
            image_embed = feats[-1]  # BX256X64X64
            high_res_feats = feats[:-1]  # BX256X64X64

            image_positional_embeddings = self.model.sam_prompt_encoder.get_dense_pe()

            if self.with_dense_prompt and dense_prompts is not None:
                _, dense_prompt_embed = self.model.sam_prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=dense_prompts,
                )
            else:
                _, dense_prompt_embed = self.model.sam_prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                )
                dense_prompt_embed = dense_prompt_embed.repeat(batch_size, 1, 1, 1)

            # SAM2 Mask Decoder
            masks, iou_pred, sam_tokens_out, object_score_logits = self.model.sam_mask_decoder(
                image_embeddings=image_embed,
                image_pe=image_positional_embeddings,
                sparse_prompt_embeddings=sparse_prompts,  # BXNX256
                dense_prompt_embeddings=dense_prompt_embed,
                multimask_output=False,
                repeat_image=False,
                high_res_features=high_res_feats
            )
        else:
            # SAM1 Encoder
            sam_visual_feat = self.model.vision_encoder(inputs)
            image_embed = sam_visual_feat['last_hidden_state']  # BX256X64X64

            image_positional_embeddings = self.model.get_image_wide_positional_embeddings()

            if self.with_dense_prompt and dense_prompts is not None:
                _, dense_prompt_embed = self.model.prompt_encoder(
                    input_points=None,
                    input_labels=None,
                    input_boxes=None,
                    input_masks=dense_prompts,
                )
            else:
                _, dense_prompt_embed = self.model.prompt_encoder(
                    input_points=None,
                    input_labels=None,
                    input_boxes=None,
                    input_masks=None,
                )
                dense_prompt_embed = dense_prompt_embed.repeat(batch_size, 1, 1, 1)

            sparse_prompts = einops.rearrange(sparse_prompts, 'b l c -> b 1 l c')
            masks, iou_predictions, mask_decoder_attentions = self.model.mask_decoder(
                image_embeddings=image_embed,
                image_positional_embeddings=image_positional_embeddings,
                sparse_prompt_embeddings=sparse_prompts,
                dense_prompt_embeddings=dense_prompt_embed,
                multimask_output=False,
                attention_similarity=None,
                target_embedding=None,
                output_attentions=None,
            )
            masks = masks.squeeze(1) # B 1 H W


        seg_mask = masks  # B 1 H W
        kwargs.update({
            'seg_mask': seg_mask,
            'dense_prompts': dense_prompts,
        })
        return kwargs


class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        num_heads: int,
        internal_dim: int = None,
        dropout: float = 0.0,
        kv_dim: int = None,
    ) -> None:
        super().__init__()
        self.query_dim = query_dim
        self.kv_dim = kv_dim if kv_dim is not None else query_dim
        self.internal_dim = internal_dim if internal_dim is not None else query_dim * 2
        self.num_heads = num_heads
        assert (
            self.internal_dim % num_heads == 0
        ), "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(query_dim, self.internal_dim)
        self.k_proj = nn.Linear(self.kv_dim, self.internal_dim)
        self.v_proj = nn.Linear(self.kv_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, query_dim)

        self.dropout_p = dropout

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor, output_vis_feat=False):
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        dropout_p = self.dropout_p if self.training else 0.0

        # Attention
        if not output_vis_feat:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
            attn_w = None
        else:
                query = q
                key = k
                value = v
                scale = None

                L, S = query.size(-2), key.size(-2)
                scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
                # attn_bias = torch.zeros(L, S, dtype=query.dtype)

                attn_weight = query @ key.transpose(-2, -1) * scale_factor
                # attn_weight += attn_bias
                attn_weight = torch.softmax(attn_weight, dim=-1)
                attn_weight = torch.dropout(attn_weight, 0.0, train=True)
                out = attn_weight @ value
                attn_w = attn_weight

        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out, attn_w


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: nn.Module = nn.ReLU,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output
        self.act = activation()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class PITCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        prompt_feat_dim: int = 0,
        img_feat_dim: int = 0,
        text_feat_dim: int = 0,
        num_heads: int = 8,
        mlp_dim: int = 1024,
        activation: nn.Module = nn.GELU,
        internal_dim: Optional[int] = None,
        attn_operation_order=None,
        is_residual: bool = True,
    ) -> None:
        super().__init__()
        assert attn_operation_order is not None, "attn_operation_order must be specified"
        assert len(attn_operation_order) == len(set(attn_operation_order)), "attn_operation_order must not contain duplicate elements"

        self.is_residual = is_residual
        self.attn_operation_order = attn_operation_order

        for attn_op in attn_operation_order:
            if 'self_attn_prompt' in attn_op:
                setattr(self, attn_op, Attention(prompt_feat_dim, num_heads, internal_dim=internal_dim))
            elif 'self_attn_text' in attn_op:
                setattr(self, attn_op, Attention(text_feat_dim, num_heads, internal_dim=internal_dim))
            elif 'cross_attn_img_text' in attn_op:
                setattr(self, attn_op, Attention(img_feat_dim, num_heads, internal_dim=internal_dim, kv_dim=text_feat_dim))
            elif 'cross_attn_prompt_text' in attn_op: 
                setattr(self, attn_op, Attention(prompt_feat_dim, num_heads, internal_dim=internal_dim, kv_dim=text_feat_dim))
            elif 'cross_attn_prompt_img' in attn_op:
                setattr(self, attn_op, Attention(prompt_feat_dim, num_heads, internal_dim=internal_dim, kv_dim=img_feat_dim))
            elif 'norm_prompt' in attn_op:
                setattr(self, attn_op, nn.LayerNorm(prompt_feat_dim))
            elif 'norm_img' in attn_op:
                setattr(self, attn_op, nn.LayerNorm(img_feat_dim))
            elif 'norm_text' in attn_op:
                setattr(self, attn_op, nn.LayerNorm(text_feat_dim))
            elif 'mlp_prompt' in attn_op:
                setattr(self, attn_op, MLP(prompt_feat_dim, mlp_dim, prompt_feat_dim, num_layers=2, activation=activation))
            elif 'mlp_img' in attn_op:
                setattr(self, attn_op, MLP(img_feat_dim, mlp_dim, img_feat_dim, num_layers=2, activation=activation))
            elif 'mlp_text' in attn_op:
                setattr(self, attn_op, MLP(text_feat_dim, mlp_dim, text_feat_dim, num_layers=2, activation=activation))
            


    def forward(self,
                text_embedding: Tensor,
                img_embedding: Tensor,
                prompt_pe: Tensor = None,
                prompts: Tensor = None,
                img_pe: Tensor = None,
                text_pe: Tensor = None,
                output_vis_feat = False,
    ):
        attn_weights = []
        for attn_op in self.attn_operation_order:
            sub_module = getattr(self, attn_op)
            img_as_query = False
            text_as_query = False
            if 'self_attn_prompt' in attn_op:
                q, k, v = prompts + prompt_pe, prompts + prompt_pe, prompts
            elif 'self_attn_text' in attn_op:
                q, k, v = text_embedding, text_embedding, text_embedding
                if text_pe is not None:
                    q = k = text_embedding + text_pe
                text_as_query = True
            elif 'cross_attn_img_text' in attn_op:
                q, k, v = img_embedding, text_embedding, text_embedding
                if img_pe is not None:
                    q = img_embedding + img_pe
                if text_pe is not None:
                    k = k + text_pe
                img_as_query = True
            elif 'cross_attn_prompt_text' in attn_op:
                q, k, v = prompts + prompt_pe, text_embedding, text_embedding
                if text_pe is not None:
                    k = text_embedding + text_pe
            elif 'cross_attn_prompt_img' in attn_op:
                q, k, v = prompts + prompt_pe, img_embedding, img_embedding
                if img_pe is not None:
                    k = img_embedding + img_pe
            elif 'norm_prompt' in attn_op:
                q = prompts
            elif 'norm_img' in attn_op:
                q = img_embedding
                img_as_query = True
            elif 'norm_text' in attn_op:
                q = text_embedding
                text_as_query = True
            elif'mlp_prompt' in attn_op:
                q = prompts
            elif'mlp_img' in attn_op:
                q = img_embedding
                img_as_query = True
            elif'mlp_text' in attn_op:
                q = text_embedding
                text_as_query = True
            else:
                raise NotImplementedError(f"Unknown attn_op: {attn_op}")
            if 'attn' in attn_op:
                out, attn_w = sub_module(q, k, v, output_vis_feat=output_vis_feat)
                attn_weights.append(attn_w)
            else:
                out = sub_module(q)
            
            if img_as_query:
                img_embedding = out + img_embedding if (self.is_residual and 'norm' not in attn_op) else out
            elif text_as_query:
                text_embedding = out + text_embedding if (self.is_residual and 'norm' not in attn_op) else out
            else:
                prompts = out + prompts if (self.is_residual and 'norm' not in attn_op) else out

        return prompts, img_embedding, text_embedding, attn_weights


class TwoQueryTextCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        query_feat_dim: int,
        text_feat_dim: int = 0,
        num_heads: int = 8,
        mlp_dim: int = 1024,
        activation: nn.Module = nn.GELU,
        internal_dim: Optional[int] = None,
        is_residual: bool = True,
        attn_operation_order=None,
    ) -> None:
        super().__init__()
        assert attn_operation_order is not None, "attn_operation_order should not be None"
        assert len(attn_operation_order) == len(set(attn_operation_order)), "attn_operation_order should not have duplicate elements"
        self.query_feat_dim = query_feat_dim
        self.text_feat_dim = text_feat_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.activation = activation
        self.internal_dim = internal_dim if internal_dim is not None else query_feat_dim * 2
        self.is_residual = is_residual
        self.attn_operation_order = attn_operation_order

        # self_attn_query_query1, cross_attn_query_text_query1_text, cross_attn_query_query_query1_query2, mlp_query1, norm_query1,
        for attn_op in attn_operation_order:
            if 'self_attn_query' in attn_op:
                setattr(self, attn_op, Attention(query_feat_dim, num_heads, internal_dim=internal_dim))
            elif 'cross_attn_query_text' in attn_op:
                setattr(self, attn_op, Attention(query_feat_dim, num_heads, internal_dim=internal_dim, kv_dim=text_feat_dim))
            elif 'cross_attn_query_query' in attn_op:
                setattr(self, attn_op, Attention(query_feat_dim, num_heads, internal_dim=internal_dim, kv_dim=query_feat_dim))
            elif 'mlp_query' in attn_op:
                setattr(self, attn_op, MLP(query_feat_dim, mlp_dim, query_feat_dim, num_layers=2, activation=activation))
            elif 'norm_query' in attn_op:
                setattr(self, attn_op, nn.LayerNorm(query_feat_dim))
            else:
                raise NotImplementedError(f"Unknown attn_op: {attn_op}")


    def forward(self,
                queries1: Tensor,
                queries2: Tensor,
                text_embedding: Tensor,
                query_pe1: Tensor=0,
                query_pe2: Tensor=0,
    ) -> Tuple[Tensor, Tensor]:
        for attn_op in self.attn_operation_order:
            sub_module = getattr(self, attn_op)
            queries1_as_query = True
            if 'self_attn_query_query1' in attn_op:
                q, k, v = queries1 + query_pe1, queries1 + query_pe1, queries1
            elif 'self_attn_query_query2' in attn_op:
                q, k, v = queries2 + query_pe2, queries2 + query_pe2, queries2
                queries1_as_query = False
            elif 'cross_attn_query_text_query1_text' in attn_op:
                q, k, v = queries1 + query_pe1, text_embedding, text_embedding
            elif 'cross_attn_query_text_query2_text' in attn_op:
                q, k, v = queries2 + query_pe2, text_embedding, text_embedding
                queries1_as_query = False
            elif 'cross_attn_query_query_query1_query2' in attn_op:
                q, k, v = queries1 + query_pe1, queries2 + query_pe2, queries2
            elif 'cross_attn_query_query_query2_query1' in attn_op:
                q, k, v = queries2 + query_pe2, queries1 + query_pe1, queries1
                queries1_as_query = False
            elif 'mlp_query1' in attn_op:
                q = queries1
            elif 'mlp_query2' in attn_op:
                q = queries2
                queries1_as_query = False
            elif 'norm_query1' in attn_op:
                q = queries1
            elif 'norm_query2' in attn_op:
                q = queries2
                queries1_as_query = False
            else:
                raise NotImplementedError(f"Unknown attn_op: {attn_op}")
            if 'attn' in attn_op:
                out, _ = sub_module(q, k, v)
            else:
                out = sub_module(q)
            if queries1_as_query:
                queries1 = queries1 + out if (self.is_residual and 'norm' not in attn_op) else out
            else:
                queries2 = queries2 + out if (self.is_residual and 'norm' not in attn_op) else out


        return queries1, queries2


@MODELS.register_module()
class CascadedPrompter(BaseModule):
    def __init__(
            self,
            num_text_queries=4,
            text_query_config=dict(
                has_pe=True,
                init_type='zero',  # zero, learnable, textpool
            ),
            prompt_config=dict(
                has_pe=True,
                init_type='zero',  # zero, learnable
            ),

            text_feat_dim=1152,
            two_queries_text_attn_depth=2,
            two_queries_text_attn_operation_order=[
                'self_attn_query_query1', 'mlp_query1_1', 'norm_query1_1', 
                'self_attn_query_query2', 'mlp_query2_1', 'norm_query2_1',
                'cross_attn_query_text_query1_text', 'mlp_query1_2', 'norm_query1_2',
                'cross_attn_query_text_query2_text', 'mlp_query2_2', 'norm_query2_2',
                'cross_attn_query_query_query1_query2', 'mlp_query1_3', 'norm_query1_3',
                'cross_attn_query_query_query2_query1','mlp_query2_3', 'norm_query2_3',
                ],

            imgpe_config=dict(type='none', size=27), # none, learnable, sine
            queries1_to_img_attn_depth=2,
            queries1_to_img_attn_operation_order=[
                'self_attn_prompt_prompt_1', 'mlp_prompt_1', 'norm_prompt_1',
                'cross_attn_img_prompt_1', 'mlp_img_1', 'norm_img_1',
            ],

            prompt_queries2_img_attn_depth=2,
            prompt_feat_dim=256,
            prompt_queries2_img_attn_operation_order=[
                'self_attn_text_text_1', 'mlp_text_1', 'norm_text_1',
                'cross_attn_img_text_1', 'mlp_img_1', 'norm_img_1',

                'self_attn_prompt_prompt_1', 'mlp_prompt_1', 'norm_prompt_1',
                'cross_attn_prompt_text_prompt_text_1', 'mlp_prompt_2', 'norm_prompt_2',
                'cross_attn_prompt_img_prompt_img_1', 'mlp_prompt_3', 'norm_prompt_3',
            ],
            num_prompts=9,
            dense_prompt_config=None,

            img_feat_dim=1152,
            num_heads=8,
            mlp_dim=512,
            internal_dim=512,
            is_residual=True,
            activation=nn.GELU,
            init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.num_text_queries = num_text_queries

        self.text_query_config = text_query_config
        if num_text_queries > 0 and text_query_config is not None:
            if text_query_config['has_pe']:
                self.text_query_pe_embedding1 = nn.Parameter(torch.randn(num_text_queries, text_feat_dim))
                self.text_query_pe_embedding2 = nn.Parameter(torch.randn(num_text_queries, text_feat_dim))
            if text_query_config['init_type'] == 'learnable':
                self.text_query_embedding1 = nn.Parameter(torch.randn(num_text_queries, text_feat_dim))
                self.text_query_embedding2 = nn.Parameter(torch.randn(num_text_queries, text_feat_dim))

        self.two_queries_text_attn_depth = two_queries_text_attn_depth
        self.two_queries_text_attn_layers = nn.ModuleList()
        for i in range(two_queries_text_attn_depth):
            self.two_queries_text_attn_layers.append(
                TwoQueryTextCrossAttentionBlock(
                    query_feat_dim=text_feat_dim,
                    text_feat_dim=text_feat_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    internal_dim=internal_dim,
                    is_residual=True,
                    attn_operation_order=two_queries_text_attn_operation_order,
                )
            )

        self.imgpe_config = imgpe_config

        if imgpe_config['type'] == 'learnable':
            self.img_pe = nn.Parameter(torch.randn(1, img_feat_dim, imgpe_config['size'], imgpe_config['size']))
        elif imgpe_config['type'] == 'sine':
            self.img_pe = SamPositionalEmbedding(mmengine.ConfigDict(
                hidden_size=img_feat_dim,
                num_pos_feats=img_feat_dim // 2,
            ))

        self.queries1_to_img_attn_layers = nn.ModuleList()
        self.queries1_to_img_attn_depth = queries1_to_img_attn_depth
        for i in range(queries1_to_img_attn_depth):
            self.queries1_to_img_attn_layers.append(
                PITCrossAttentionBlock(
                    text_feat_dim=text_feat_dim,
                    img_feat_dim=img_feat_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    internal_dim=internal_dim,
                    attn_operation_order=queries1_to_img_attn_operation_order,
                    is_residual=True,
                )
            )

        self.prompt_queries2_img_attn_layers = nn.ModuleList()
        for i in range(prompt_queries2_img_attn_depth):
            self.prompt_queries2_img_attn_layers.append(
                PITCrossAttentionBlock(
                    prompt_feat_dim=prompt_feat_dim,
                    img_feat_dim=img_feat_dim,
                    text_feat_dim=text_feat_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    internal_dim=internal_dim,
                    attn_operation_order=prompt_queries2_img_attn_operation_order,
                )
            )

        self.prompt_config = prompt_config
        self.dense_prompt_config = dense_prompt_config
        num_prompts = num_prompts + 1 if (dense_prompt_config is not None and 'prompt' == dense_prompt_config['proj_from']) else num_prompts
        if prompt_config['has_pe']:
            self.prompt_pe_embedding = nn.Parameter(torch.randn(num_prompts, prompt_feat_dim))
        if prompt_config['init_type'] == 'learnable':
            self.prompt_embedding = nn.Parameter(torch.randn(num_prompts, prompt_feat_dim))

        self.sparse_prompt_mlp = MLP(prompt_feat_dim, mlp_dim, prompt_feat_dim, num_layers=2, activation=activation)

        if dense_prompt_config is not None and 'text_queries2_pool' in dense_prompt_config['proj_from']:
            self.dense_prompt_pool_mlp_1 = nn.Sequential(
                nn.Linear(text_feat_dim, text_feat_dim),
                nn.GELU())
            self.dense_prompt_pool_mlp_2 = nn.Sequential(
                nn.Conv1d(text_feat_dim, text_feat_dim, kernel_size=num_text_queries, stride=1, padding=0, bias=True),
                nn.BatchNorm1d(text_feat_dim),
                nn.GELU(),
                nn.Conv1d(text_feat_dim, text_feat_dim, kernel_size=1, stride=1, padding=0, bias=True),
            )

        if dense_prompt_config is not None and dense_prompt_config.get('has_internal_proj', False):
            input_dim = prompt_feat_dim if dense_prompt_config['proj_from'] == 'prompt' else text_feat_dim
            self.dense_prompt_mlp = MLP(input_dim, mlp_dim, img_feat_dim, num_layers=2, activation=activation)

        if dense_prompt_config is not None and dense_prompt_config.get('has_internal_conv', False):
            self.dense_prompt_conv = nn.Sequential(
                nn.Conv2d(img_feat_dim, img_feat_dim // 2, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(img_feat_dim // 2),
                nn.GELU(),
                nn.Conv2d(img_feat_dim // 2, img_feat_dim, kernel_size=3, stride=1, padding=1, bias=True),
            )

    def init_weights(self):
        pass

    def get_image_positional_embeddings(self, size, sam_model):

        target_device = self.img_pe.positional_embedding.device
        target_dtype = self.img_pe.positional_embedding.dtype
        grid = torch.ones((size, size), device=target_device, dtype=target_dtype)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / size
        x_embed = x_embed / size
        positional_embedding = self.img_pe(torch.stack([x_embed, y_embed], dim=-1))
        positional_embedding = positional_embedding.permute(2, 0, 1).unsqueeze(0)  # channel x height x width
        return positional_embedding  # 1 x channel x height x width


    def forward(
            self,
            clip_visual_feat: Tensor, # BX(HXW=729)X1152
            clip_text_feat: Tensor,  # 4X11X1152
            clip_text_feat_pooler: Tensor,  # 4X1152
            clip_text_encoder=None,
            sam_model=None,  # sam_model
            output_vis_feat=False,
    ):

        clip_pool_text = clip_text_feat_pooler
        clip_visual_feat_map = einops.rearrange(clip_visual_feat, 'b (h w) c -> b c h w', h=int(math.sqrt(clip_visual_feat.shape[1])))  # BX1152X27X27
        batch_size = clip_visual_feat.size(0)  # (bs, num_feat_points, dim)
        if output_vis_feat:
            output_vis_feats = {}
            attn_weights_list = []
            output_vis_feats['clip_pool_text'] = clip_pool_text.detach().cpu().clone()
            output_vis_feats['clip_visual_feat_map'] = clip_visual_feat_map.detach().cpu().clone()

        if self.two_queries_text_attn_depth > 0:
            if self.text_query_config['has_pe']:
                text_queries_pe1 = einops.repeat(self.text_query_pe_embedding1, 'n d -> b n d', b=batch_size)
                text_queries_pe2 = einops.repeat(self.text_query_pe_embedding2, 'n d -> b n d', b=batch_size)
            else:
                text_queries_pe1 = 0
                text_queries_pe2 = 0

            if self.text_query_config['init_type'] == 'learnable':
                text_queries1 = einops.repeat(self.text_query_embedding1, 'n d -> b n d', b=batch_size)
                text_queries2 = einops.repeat(self.text_query_embedding2, 'n d -> b n d', b=batch_size)
            elif self.text_query_config['init_type'] == 'textpool':
                # B L C -> B num_text_queries C
                text_queries1 = F.adaptive_avg_pool1d(clip_text_feat.permute(0, 2, 1), self.num_text_queries).permute(0, 2, 1)
                text_queries2 = text_queries1
            elif self.text_query_config['init_type'] == 'zero':
                text_queries1 = torch.zeros_like(text_queries_pe1)
                text_queries2 = torch.zeros_like(text_queries_pe2)
            else:
                raise NotImplementedError(f"Unknown text_query_init_type: {self.text_query_config['init_type']}")

            for layer in self.two_queries_text_attn_layers:
                text_queries1, text_queries2 = layer(
                    queries1 = text_queries1,
                    queries2 = text_queries2,
                    text_embedding=clip_text_feat,
                    query_pe1=text_queries_pe1,
                    query_pe2=text_queries_pe2,
                )
        else:
            text_queries_pe1 = None
            text_queries_pe2 = None
            text_queries1 = clip_text_feat
            text_queries2 = clip_text_feat

        if output_vis_feat:
            output_vis_feats['text_queries1'] = text_queries1.detach().cpu().clone()
            output_vis_feats['text_queries2'] = text_queries2.detach().cpu().clone()

        if self.imgpe_config['type'] == 'learnable':
            img_pe = self.img_pe
            if img_pe.size(2)!= clip_visual_feat_map.size(2):
                print(f"Resize img_pe from {img_pe.size(2)} to {clip_visual_feat_map.size(2)}")
                img_pe = F.interpolate(img_pe, size=(clip_visual_feat_map.size(2), clip_visual_feat_map.size(2)), mode='bicubic', align_corners=True)
        elif self.imgpe_config['type'] =='sine':
            img_pe = self.get_image_positional_embeddings(clip_visual_feat_map.size(2), sam_model)
        else:
            img_pe = None
        if img_pe is not None:
            img_pe = einops.rearrange(img_pe, 'b c h w -> b (h w) c')

        # step 1
        if self.queries1_to_img_attn_depth > 0:
            refined_img_embedding = clip_visual_feat
            refined_text_queries1 = text_queries1
            for layer in self.queries1_to_img_attn_layers:
                _, refined_img_embedding, refined_text_queries1, attn_weights = layer(
                    prompts=None,
                    img_embedding=refined_img_embedding,
                    text_embedding=refined_text_queries1,
                    img_pe=img_pe,
                    prompt_pe=None,
                    text_pe=text_queries_pe1,
                    output_vis_feat=output_vis_feat
                )
        else:
            refined_img_embedding = clip_visual_feat
            refined_text_queries1 = text_queries1

        if output_vis_feat:
            attn_weights_list += attn_weights
            output_vis_feats['refined_img_embedding1'] = refined_img_embedding.detach().cpu().clone()
            output_vis_feats['refined_text_queries1'] = refined_text_queries1.detach().cpu().clone()

        # step 2
        if self.prompt_config['has_pe']:
            prompt_pe = einops.repeat(self.prompt_pe_embedding, 'n d -> b n d', b=batch_size)
        else:
            prompt_pe = 0
        if self.prompt_config['init_type'] == 'learnable':
            prompt = einops.repeat(self.prompt_embedding, 'n d -> b n d', b=batch_size)
        else:
            prompt = torch.zeros_like(prompt_pe)
        refined_text_queries2 = text_queries2

        for layer in self.prompt_queries2_img_attn_layers:
            prompt, refined_img_embedding, refined_text_queries2, attn_weights = layer(
                prompts=prompt,
                text_embedding=refined_text_queries2,

                img_embedding=refined_img_embedding,
                img_pe=img_pe,
                prompt_pe=prompt_pe,
                text_pe=text_queries_pe2,

                output_vis_feat=output_vis_feat
            )
        if output_vis_feat:
            attn_weights_list += attn_weights
            output_vis_feats['refined_img_embedding2'] = refined_img_embedding.detach().cpu().clone()
            output_vis_feats['refined_text_queries2'] = refined_text_queries2.detach().cpu().clone()
        else:
            output_vis_feats = None

        if self.dense_prompt_config is not None and 'prompt' == self.dense_prompt_config['proj_from']:
            dense_prompts = prompt[:, 0, :]
            sparse_prompts = prompt[:, 1:, :]
        else:
            dense_prompts = None
            sparse_prompts = prompt

        sparse_prompts = self.sparse_prompt_mlp(sparse_prompts)

        if self.dense_prompt_config is not None and 'text_pool' == self.dense_prompt_config['proj_from']:
            dense_prompts = clip_pool_text
        elif self.dense_prompt_config is not None and 'text_queries2_pool' in self.dense_prompt_config['proj_from']:
            dense_prompts = text_queries2 if self.dense_prompt_config['proj_from'] == 'text_queries2_pool' else refined_text_queries2
            # text_queries2 B N C
            dense_prompts = self.dense_prompt_pool_mlp_1(dense_prompts)
            dense_prompts = self.dense_prompt_pool_mlp_2(dense_prompts.permute(0, 2, 1))
            dense_prompts = dense_prompts.squeeze(-1)

        if self.dense_prompt_config is not None and self.dense_prompt_config.get('has_internal_proj', False):
            dense_prompts = self.dense_prompt_mlp(dense_prompts)

        # for sam1
        if hasattr(sam_model.model, 'prompt_encoder'):
            mask_size = sam_model.model.prompt_encoder.input_image_size // 4
        else:
            mask_size = sam_model.model.sam_prompt_encoder.mask_input_size[0]
        refined_img_feat_map = einops.rearrange(refined_img_embedding, 'b (h w) c -> b c h w', h=int(math.sqrt(refined_img_embedding.shape[1])))

        if dense_prompts is not None:
            if 'clipfeat' == self.dense_prompt_config.get('featmap_from', ''):
                visual_feat_map = clip_visual_feat_map
            elif 'refined_clipfeat' == self.dense_prompt_config.get('featmap_from', ''):
                visual_feat_map = refined_img_feat_map
            else:
                raise NotImplementedError(f"Unknown dense_prompt_config: {self.dense_prompt_config}")

            if self.dense_prompt_config.get('has_internal_conv', False):
                visual_feat_map = self.dense_prompt_conv(visual_feat_map)

        if dense_prompts is not None:
            if 'preup' in self.dense_prompt_config['up_strategy']:
                visual_feat_map = F.interpolate(visual_feat_map, size=(mask_size, mask_size), mode='bilinear', align_corners=True)
            normalized_clip_visual_feat = visual_feat_map / visual_feat_map.norm(p=2, dim=1, keepdim=True)
            dense_prompts = dense_prompts / dense_prompts.norm(p=2, dim=1, keepdim=True)
            dense_prompts = einops.einsum(normalized_clip_visual_feat, dense_prompts, 'b c h w, b c -> b h w').unsqueeze(1)
            dense_prompts = dense_prompts * clip_text_encoder.logit_scale.exp() + clip_text_encoder.logit_bias
            if 'postup' in self.dense_prompt_config['up_strategy']:
                dense_prompts = F.interpolate(dense_prompts, size=(mask_size, mask_size), mode='bilinear', align_corners=True)
        if output_vis_feat:
            # output_vis_feats['clip_pool_text'] = clip_pool_text.cpu()
            # output_vis_feats['clip_visual_feat_map'] = clip_visual_feat_map.cpu()
            # output_vis_feats['text_queries1'] = text_queries1.cpu()
            # output_vis_feats['text_queries2'] = text_queries2.cpu()
            # output_vis_feats['refined_img_embedding1'] = refined_img_embedding.cpu()
            # output_vis_feats['refined_text_queries1'] = refined_text_queries1
            # output_vis_feats['refined_img_embedding2'] = refined_img_embedding.cpu()
            # output_vis_feats['refined_text_queries2'] = refined_text_queries2

            clip_pool_text_0 = output_vis_feats['clip_pool_text']
            clip_visual_feat_map_0 = output_vis_feats['clip_visual_feat_map']
            text_queries1_0 = output_vis_feats['text_queries1']
            text_queries2_0 = output_vis_feats['text_queries2']
            refined_img_embedding1_0 = output_vis_feats['refined_img_embedding1']
            refined_img_embedding2_0 = output_vis_feats['refined_img_embedding2']
            refined_text_queries1_0 = output_vis_feats['refined_text_queries1']
            refined_text_queries2_0 = output_vis_feats['refined_text_queries2']

            clip_pool_text_0 = clip_pool_text_0 / clip_pool_text_0.norm(p=2, dim=-1, keepdim=True)
            clip_visual_feat_map_0 = clip_visual_feat_map_0 / clip_visual_feat_map_0.norm(p=2, dim=1, keepdim=True)

            text_queries1_0 = text_queries1_0.mean(dim=1)
            text_queries1_0 = text_queries1_0 / text_queries1_0.norm(dim=-1, keepdim=True)

            text_queries2_0 = text_queries2_0.mean(dim=1)
            text_queries2_0 = text_queries2_0 / text_queries2_0.norm(dim=-1, keepdim=True)

            refined_text_queries1_0 = refined_text_queries1_0.mean(dim=1)
            refined_text_queries1_0 = refined_text_queries1_0 / refined_text_queries1_0.norm(dim=-1, keepdim=True)

            refined_text_queries2_0 = refined_text_queries2_0.mean(dim=1)
            refined_text_queries2_0 = refined_text_queries2_0 / refined_text_queries2_0.norm(dim=-1, keepdim=True)


            refined_img_embedding1_0 = einops.rearrange(refined_img_embedding1_0, 'b (h w) c -> b c h w', h=int(math.sqrt(refined_img_embedding1_0.shape[1])))
            refined_img_embedding2_0 = einops.rearrange(refined_img_embedding2_0, 'b (h w) c -> b c h w', h=int(math.sqrt(refined_img_embedding2_0.shape[1])))
            refined_img_embedding1_0 = refined_img_embedding1_0 / refined_img_embedding1_0.norm(dim=1, keepdim=True)
            refined_img_embedding2_0 = refined_img_embedding2_0 / refined_img_embedding2_0.norm(dim=1, keepdim=True)

            return_dict = {'dense_prompts': dense_prompts.detach().cpu().squeeze()}

            for v1, k1 in zip([clip_pool_text_0, text_queries1_0, text_queries2_0, refined_text_queries1_0, refined_text_queries2_0], ['t', 'q1', 'q2', 'refined_q1', 'refined_q2']):
                for v2, k2 in zip([clip_visual_feat_map_0, refined_img_embedding1_0, refined_img_embedding2_0], ['v', 'v1', 'v2']):
                    return_dict[f'{k1}_{k2}'] = einops.einsum(v2, v1, 'b c h w, b c -> b h w').detach().cpu()
            return_dict['q1_i'] = attn_weights_list[1].detach().cpu().clone()
            return_dict['q2_i'] = attn_weights_list[3].detach().cpu().clone()
            return_dict['p_i'] = attn_weights_list[6].detach().cpu().clone()
            return_dict['r_i_0'] = clip_visual_feat_map_0
            return_dict['r_i_1'] = refined_img_embedding1_0
            return_dict['r_i_2'] = refined_img_embedding2_0


        return {
            'sparse_prompts': sparse_prompts,
            'dense_prompts': dense_prompts,
            'text_queries1': text_queries1,
            'text_queries2': text_queries2,
            'clip_pool_text': clip_pool_text,
            'refined_img_feat_map': refined_img_feat_map,
            'clip_visual_feat_map': clip_visual_feat_map,
            'output_vis_feats': return_dict if output_vis_feat else None,
        }


@MODELS.register_module()
class RefSegSiglipVisionModel(BaseModule):
    def __init__(
            self,
            model_name_or_path: str='thomas/siglip-so400m-patch14-384',
            cache_dir=None,
            init_cfg=None,
            input_size=384,
    ):
        super().__init__(init_cfg=init_cfg)
        self.model_name_or_path = model_name_or_path
        self.processor = AutoProcessor.from_pretrained(model_name_or_path, cache_dir=cache_dir).image_processor
        self.model = AutoModel.from_pretrained(model_name_or_path, cache_dir=cache_dir).vision_model
        self.config = self.model.config
        self.model.is_init = True
        ori_image_size = self.config.image_size[0] if isinstance(self.config.image_size, Sequence) else self.config.image_size
        self.interpolate_pos_encoding = input_size is not None and input_size != ori_image_size

    def init_weights(self):
        pass

    def forward(self, *args, **kwargs):
        return self.model(*args, interpolate_pos_encoding=self.interpolate_pos_encoding, **kwargs)

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

@MODELS.register_module()
class PseudoSegHead(BaseModule):
    def __init__(
            self,
            num_classes: int=2,
            out_channels: int=1,
            threshold: float=0.5,
            loss_decode=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0),
            dst_size=27,
            ignore_index=255,
            sampler=None,
            align_corners=False,
            extra_loss=dict(),
            init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)

        self.num_classes = num_classes
        self.out_channels = out_channels
        self.threshold = threshold
        self.ignore_index = ignore_index
        self.align_corners = align_corners

        if isinstance(loss_decode, dict):
            self.loss_decode = MODELS.build(loss_decode)
        elif isinstance(loss_decode, (list, tuple)):
            self.loss_decode = nn.ModuleList()
            for loss in loss_decode:
                self.loss_decode.append(MODELS.build(loss))
        else:
            raise TypeError(f'loss_decode must be a dict or sequence of dict,\
                but got {type(loss_decode)}')

        if sampler is not None:
            self.sampler = build_pixel_sampler(sampler, context=self)
        else:
            self.sampler = None

        self.MIL_loss = MILCrossEntropy()
        self.extra_loss = extra_loss
        self.logit_scale1 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale2 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_bias1 = nn.Parameter(torch.ones([]) * np.log(0.5))
        self.logit_bias2 = nn.Parameter(torch.ones([]) * np.log(0.5))
        self.dst_size = (dst_size, dst_size) if isinstance(dst_size, int) else dst_size

    def forward(self, inputs):
        return inputs

    def loss(self, inputs, batch_data_samples: SampleList, train_cfg: ConfigType) -> dict:
        seg_mask = inputs.pop('seg_mask')
        seg_logits = self.forward(seg_mask)
        losses = self.loss_by_feat(seg_logits, inputs, batch_data_samples)
        return losses

    def predict(self, inputs: Tuple[Tensor], batch_img_metas: List[dict], test_cfg: ConfigType) -> Tensor:
        seg_mask = inputs['seg_mask']
        seg_logits = self.forward(seg_mask)
        return self.predict_by_feat(seg_logits, batch_img_metas)

    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
        gt_semantic_segs = [
            data_sample.gt_sem_seg.data for data_sample in batch_data_samples
        ]
        return torch.stack(gt_semantic_segs, dim=0)

    def gather_features(
            self,
            features,
            local_loss=False,
            gather_with_grad=False,
            rank=0,
            world_size=1,
            use_horovod=False
    ):
        assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
        if use_horovod:
            assert hvd is not None, 'Please install horovod'
            if gather_with_grad:
                all_features = hvd.allgather(features)
            else:
                with torch.no_grad():
                    all_features = hvd.allgather(features)
                if not local_loss:
                    # ensure grads for local rank when all_* features don't have a gradient
                    gathered_features = list(all_features.chunk(world_size, dim=0))
                    gathered_features[rank] = features
                    all_features = torch.cat(gathered_features, dim=0)
        else:
            # We gather tensors from all gpus
            if gather_with_grad:
                all_features = torch.cat(torch.distributed.nn.all_gather(features), dim=0)
            else:
                gathered_features = [torch.zeros_like(features) for _ in range(world_size)]
                dist.all_gather(gathered_features, features)
                if not local_loss:
                    # ensure grads for local rank when all_* features don't have a gradient
                    gathered_features[rank] = features
                all_features = torch.cat(gathered_features, dim=0)

        return all_features

    def loss_by_feat(self, 
        seg_logits: Tensor,
        inputs: dict,
        batch_data_samples: SampleList) -> dict:
        
        seg_label = self._stack_batch_gt(batch_data_samples)
        loss = dict()
        seg_logits = resize(
            input=seg_logits,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logits, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)

        dense_prompts = inputs['dense_prompts'] # 16X1X256X256
        text_queries1 = inputs['text_queries1']  # bx3x1152
        text_queries2 = inputs['text_queries2']  # bx3x1152
        clip_pool_text = inputs['clip_pool_text']  # bx1152


        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode

        for loss_decode in losses_decode:
            loss[loss_decode.loss_name] = loss_decode(
                seg_logits,
                seg_label,
                weight=seg_weight,
                ignore_index=self.ignore_index)
            
            loss_weight = self.extra_loss.get('dense_prompts', None)
            if loss_weight is not None:
                dense_prompts = resize(
                    input=dense_prompts,
                    size=seg_label.shape[1:],
                    mode='bilinear',
                    align_corners=self.align_corners)
                loss[loss_decode.loss_name+'_dense_prompts'] = loss_weight * loss_decode(
                    dense_prompts,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)


        loss_weight = self.extra_loss.get('text_queries1_queries2', None)
        if loss_weight is not None:
            # should push text_queries1 and text_queries2 away from each other using cosine similarity
            text_queries1_mean = torch.mean(text_queries1, dim=1)
            text_queries2_mean = torch.mean(text_queries2, dim=1)
            text_queries1_mean_nrom = F.normalize(text_queries1_mean, dim=1)
            text_queries2_mean_nrom = F.normalize(text_queries2_mean, dim=1)
            similarity = F.cosine_similarity(text_queries1_mean_nrom, text_queries2_mean_nrom, dim=1)
            loss['loss_text_queries1_queries2'] = loss_weight * torch.mean(similarity**2)

        visual_feat_map = None
        if 'refined_img_feat_map_perimg' in self.extra_loss or 'refined_img_feat_map_perbatch' in self.extra_loss:
            visual_feat_map = inputs['refined_img_feat_map']  # bX1152X256X256
        elif 'clip_visual_feat_map_perimg' in self.extra_loss or 'clip_visual_feat_map_perbatch' in self.extra_loss:
            visual_feat_map = inputs['clip_visual_feat_map']  # bX1152X27X27


        if visual_feat_map is not None:
            visual_feat_map = F.interpolate(visual_feat_map, size=self.dst_size, mode='bilinear', align_corners=True)
            seg_label_downsample = (F.interpolate(seg_label.unsqueeze(1).float(), size=self.dst_size, mode='bilinear', align_corners=True).squeeze(1) > 0.5).long()

        loss_weight = self.extra_loss.get('refined_img_feat_map_perimg', None) or self.extra_loss.get('clip_visual_feat_map_perimg', None)
        if loss_weight is not None:
            logit_scale1 = self.logit_scale1.exp()
            refined_img_feat_map_norm = F.normalize(visual_feat_map, dim=1)
            if self.extra_loss.get('use_clip_pool_text', False):
                text_queries2_mean_norm = F.normalize(clip_pool_text, dim=1)
            elif self.extra_loss.get('use_text_queries2', False):
                text_queries2_mean_norm = F.normalize(torch.mean(text_queries2, dim=1), dim=1)
            else:
                raise NotImplementedError(f"Unknown use_text_queries2: {self.extra_loss}")
            refined_img_feat_map_query2 = logit_scale1 * einops.einsum(refined_img_feat_map_norm, text_queries2_mean_norm, 'b c h w, b c -> b h w') + self.logit_bias1
            if self.extra_loss.get('refined_img_feat_map_perimg_ce', False):
                for loss_decode in losses_decode:
                    loss[loss_decode.loss_name+'_refined_img_feat_map_perimg'] = loss_weight * loss_decode(
                        refined_img_feat_map_query2,
                        seg_label_downsample,
                        weight=seg_weight,
                        ignore_index=self.ignore_index)
            elif self.extra_loss.get('refined_img_feat_map_perimg_mil', False):
                loss['loss_mil_refined_img_feat_map_perimg'] = loss_weight * self.MIL_loss(refined_img_feat_map_query2, seg_label_downsample, dim=-1)
            else:
                raise NotImplementedError(f"Unknown refined_img_feat_map_perimg: {self.extra_loss}")

        loss_weight = self.extra_loss.get('refined_img_feat_map_perbatch', None) or self.extra_loss.get('clip_visual_feat_map_perbatch', None)
        if loss_weight is not None:
            logit_scale2 = self.logit_scale2.exp()
            # pooling the refined_img_feat_map by seg_label
            refined_img_feat_map_mask = einops.einsum(visual_feat_map, seg_label_downsample, 'b c h w, b h w -> b c h w')
            refined_img_vector = torch.mean(refined_img_feat_map_mask, dim=(2,3))
            refined_img_vector_norm = F.normalize(refined_img_vector, dim=1)
            text_queries2_mean_nrom = F.normalize(torch.mean(text_queries2, dim=1), dim=1)

            rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
            local_loss = False
            if world_size > 1:
                all_refined_img_vector_norm = self.gather_features(refined_img_vector_norm, local_loss=local_loss, gather_with_grad=False, rank=rank, world_size=world_size)
                all_text_queries2_mean_nrom = self.gather_features(text_queries2_mean_nrom, local_loss=local_loss, gather_with_grad=False, rank=rank, world_size=world_size)

                if local_loss:
                    logits_per_image = logit_scale2 * refined_img_vector_norm @ all_text_queries2_mean_nrom.t()
                    logits_per_text = logit_scale2 * text_queries2_mean_nrom @ all_refined_img_vector_norm.t()
                else:
                    logits_per_image = logit_scale2 * all_refined_img_vector_norm @ all_text_queries2_mean_nrom.t()
                    logits_per_text = logits_per_image.T
            else:
                logits_per_image = logit_scale2 * refined_img_vector_norm @ text_queries2_mean_nrom.t()
                logits_per_text = logit_scale2 * text_queries2_mean_nrom @ refined_img_vector_norm.t()

            logits_per_image = logits_per_image + self.logit_bias2
            logits_per_text = logits_per_text + self.logit_bias2

            device = refined_img_vector_norm.device
            num_logits = logits_per_image.shape[0]
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if world_size > 1 and local_loss:
                labels = labels + num_logits * rank

            loss_perbatch = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2
            loss['loss_perbatch_refined_img_feat_map_perbatch'] = loss_weight * loss_perbatch

        loss['acc_seg'] = accuracy(seg_logits, seg_label, ignore_index=self.ignore_index)
        return loss

    def predict_by_feat(self, seg_logits: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        if isinstance(batch_img_metas[0]['img_shape'], torch.Size):
            # slide inference
            size = batch_img_metas[0]['img_shape']
        elif 'pad_shape' in batch_img_metas[0]:
            size = batch_img_metas[0]['pad_shape'][:2]
        else:
            size = batch_img_metas[0]['img_shape']

        seg_logits = resize(
            input=seg_logits,
            size=size,
            mode='bilinear',
            align_corners=self.align_corners)
        return seg_logits