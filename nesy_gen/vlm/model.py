import torch
import torch.nn as nn
import torchvision
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

class VisionT5(nn.Module):
    def __init__(
        self,
        text_model_name: str = "t5-small",
        visual_backbone: str = "densenet121",
        freeze_visual_encoder: bool = True,
        use_diagnosis_prompts: bool = False,
        cls_lambda: float = 0.5
    ):
        super().__init__()
        self.freeze_visual_encoder = freeze_visual_encoder
        self.visual_backbone = visual_backbone
        self.use_diagnosis_prompts = use_diagnosis_prompts
        self.cls_lambda = cls_lambda
        self.num_chexpert_labels = 14
        
        # Initialize visual backbone features
        self._init_visual_encoder(visual_backbone)
                
        # Load T5 model
        self.t5 = T5ForConditionalGeneration.from_pretrained(text_model_name)
        self.d_model = self.t5.config.d_model
        
        # Projection layer: maps visual features to T5 embedding space (d_model)
        self.proj = nn.Linear(self.num_visual_features, self.d_model)
        
        # Classification layer (fully initialized to maintain backward compatibility)
        self.classifier = nn.Linear(self.num_visual_features, self.num_chexpert_labels)
        self._cls_criterion = nn.BCEWithLogitsLoss()

    def _init_visual_encoder(self, visual_backbone: str):
        self.is_swin = False
        self.is_vit = False

        if visual_backbone == "densenet121":
            self.visual_encoder = torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.DEFAULT)
            self.visual_features = self.visual_encoder.features
            self.num_visual_features = 1024
        elif visual_backbone == "resnet18":
            self.visual_encoder = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
            self.visual_features = nn.Sequential(*list(self.visual_encoder.children())[:-2])
            self.num_visual_features = 512
        elif visual_backbone == "resnet50":
            self.visual_encoder = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
            self.visual_features = nn.Sequential(*list(self.visual_encoder.children())[:-2])
            self.num_visual_features = 2048
        elif visual_backbone == "efficientnet_b0":
            self.visual_encoder = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT)
            self.visual_features = self.visual_encoder.features
            self.num_visual_features = 1280
        elif visual_backbone == "efficientnet_b4":
            self.visual_encoder = torchvision.models.efficientnet_b4(weights=torchvision.models.EfficientNet_B4_Weights.DEFAULT)
            self.visual_features = self.visual_encoder.features
            self.num_visual_features = 1792
        elif visual_backbone in ["swin_tiny", "swin_base", "vit_base_patch16_224"]:
            try:
                import timm
            except ImportError:
                raise ImportError("timm is required for Swin/ViT backbones: pip install timm")
            
            if visual_backbone == "swin_tiny":
                self.visual_features = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, num_classes=0, global_pool="")
                self.num_visual_features = 768
                self.is_swin = True
            elif visual_backbone == "swin_base":
                self.visual_features = timm.create_model("swin_base_patch4_window7_224", pretrained=True, num_classes=0, global_pool="")
                self.num_visual_features = 1024
                self.is_swin = True
            elif visual_backbone == "vit_base_patch16_224":
                self.visual_features = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0, global_pool="")
                self.num_visual_features = 768
                self.is_vit = True
        else:
            raise ValueError(f"Unsupported visual backbone: {visual_backbone}")
            
        if self.freeze_visual_encoder:
            for param in self.visual_features.parameters():
                param.requires_grad = False
        else:
            # Partial fine-tuning for Swin Transformers
            if hasattr(self.visual_features, "patch_embed"):
                for param in self.visual_features.patch_embed.parameters():
                    param.requires_grad = False
            if hasattr(self.visual_features, "layers"):
                for i in range(min(2, len(self.visual_features.layers))):
                    for param in self.visual_features.layers[i].parameters():
                        param.requires_grad = False
        
    def _extract_visual_features(self, images):
        is_timm = self.is_swin or self.is_vit
        
        if self.freeze_visual_encoder:
            self.visual_features.eval()
            with torch.no_grad():
                if is_timm:
                    raw = self.visual_features.forward_features(images)
                else:
                    raw = self.visual_features(images)
        else:
            if is_timm:
                raw = self.visual_features.forward_features(images)
            else:
                raw = self.visual_features(images)
            
        if self.is_swin:
            if raw.dim() == 4:  # (B, H, W, C)
                B, H, W, C = raw.shape
                spatial_feats = raw.reshape(B, H * W, C)
            else:
                spatial_feats = raw
        elif self.is_vit:
            spatial_feats = raw[:, 1:, :]  # Exclude CLS token
        else:
            # CNN output: (B, C, H, W) -> flatten to (B, H*W, C)
            spatial_feats = raw.flatten(2).transpose(1, 2)
            
        pooled_feats = spatial_feats.mean(dim=1)  # (B, C)
        return spatial_feats, pooled_feats

    def forward(
        self,
        images,
        decoder_input_ids=None,
        labels=None,
        encoder_input_ids=None,
        encoder_attention_mask=None,
        chexpert_labels=None
    ):
        spatial_feats, pooled_feats = self._extract_visual_features(images)
        batch_size = spatial_feats.size(0)
        device = images.device
        
        # ── Classification head (active only when use_diagnosis_prompts=True) ──
        cls_loss = torch.tensor(0.0, device=device)
        if self.use_diagnosis_prompts and chexpert_labels is not None:
            cls_logits = self.classifier(pooled_feats)
            cls_loss = self._cls_criterion(cls_logits, chexpert_labels.float().to(device))
            
        # Project visual features to T5 embedding space
        visual_embeds = self.proj(spatial_feats)  # (B, N, d_model)
        vis_mask = torch.ones((batch_size, visual_embeds.size(1)), dtype=torch.long, device=device)
        
        if encoder_input_ids is not None:
            # Encode text tokens only (Stable July 8 setup)
            text_encoder_outputs = self.t5.encoder(
                input_ids=encoder_input_ids,
                attention_mask=encoder_attention_mask
            )
            text_feats = text_encoder_outputs.last_hidden_state  # (B, L, d_model)
            
            # Combine visual and text tokens after the encoder (Stable cross-attention alignment)
            combined_tokens = torch.cat([visual_embeds, text_feats], dim=1)  # (B, N + L, d_model)
            combined_mask = (
                torch.cat([vis_mask, encoder_attention_mask], dim=1)
                if encoder_attention_mask is not None
                else vis_mask
            )
        else:
            combined_tokens = visual_embeds
            combined_mask = vis_mask
            
        encoder_outputs = BaseModelOutput(last_hidden_state=combined_tokens)
        
        # Shift right manually to calculate manual label-smoothed cross-entropy loss
        if decoder_input_ids is None and labels is not None:
            decoder_input_ids = self.t5._shift_right(labels)
            
        t5_out = self.t5(
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            attention_mask=combined_mask
        )
        
        logits = t5_out.logits
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=-100)
            gen_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.to(device).view(-1))
            
            # Combine generation loss and classification loss
            t5_out.loss = gen_loss + self.cls_lambda * cls_loss
            t5_out.gen_loss = gen_loss
            t5_out.cls_loss = cls_loss
        else:
            t5_out.loss = cls_loss
            t5_out.gen_loss = torch.tensor(0.0, device=device)
            t5_out.cls_loss = cls_loss
            
        return t5_out

    def generate(self, images, encoder_input_ids=None, encoder_attention_mask=None, **kwargs):
        # Pop custom parameters that are not accepted by HuggingFace T5 generate
        kwargs.pop("tokenizer", None)
        kwargs.pop("template_reports", None)
        
        with torch.no_grad():
            spatial_feats, pooled_feats = self._extract_visual_features(images)
            batch_size = spatial_feats.size(0)
            device = images.device
            
            visual_embeds = self.proj(spatial_feats)
            vis_mask = torch.ones((batch_size, visual_embeds.size(1)), dtype=torch.long, device=device)
            
            if encoder_input_ids is not None:
                text_encoder_outputs = self.t5.encoder(
                    input_ids=encoder_input_ids,
                    attention_mask=encoder_attention_mask
                )
                text_feats = text_encoder_outputs.last_hidden_state
                combined_tokens = torch.cat([visual_embeds, text_feats], dim=1)
                combined_mask = (
                    torch.cat([vis_mask, encoder_attention_mask], dim=1)
                    if encoder_attention_mask is not None
                    else vis_mask
                )
            else:
                combined_tokens = visual_embeds
                combined_mask = vis_mask
                
            encoder_outputs = BaseModelOutput(last_hidden_state=combined_tokens)
            
            return self.t5.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=combined_mask,
                **kwargs
            )
            
    def save_checkpoint(self, path: str):
        import json
        from pathlib import Path
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        
        # Save PyTorch states
        torch.save({
            "proj_state_dict": self.proj.state_dict(),
            "classifier_state_dict": self.classifier.state_dict(),
            "visual_features_state_dict": self.visual_features.state_dict() if not self.freeze_visual_encoder else None
        }, p / "vision_t5_weights.bin")
        
        # Save T5 model checkpoint
        self.t5.save_pretrained(p / "text_model")
        
        # Save custom config
        config = {
            "d_model": self.d_model,
            "freeze_visual_encoder": self.freeze_visual_encoder,
            "num_visual_features": self.num_visual_features,
            "visual_backbone": self.visual_backbone,
            "use_diagnosis_prompts": self.use_diagnosis_prompts,
            "cls_lambda": self.cls_lambda,
            "text_model_name": self.t5.config._name_or_path if hasattr(self.t5.config, "_name_or_path") else "t5-small"
        }
        with open(p / "r2gen_t5_config.json", "w") as f:
            json.dump(config, f)
            
    def load_checkpoint(self, path: str):
        import json
        from pathlib import Path
        p = Path(path)
        
        # Load config to configure backbone dynamically
        with open(p / "r2gen_t5_config.json", "r") as f:
            config = json.load(f)
            
        self.freeze_visual_encoder = config.get("freeze_visual_encoder", self.freeze_visual_encoder)
        self.visual_backbone = config.get("visual_backbone", "densenet121")
        self.use_diagnosis_prompts = config.get("use_diagnosis_prompts", self.use_diagnosis_prompts)
        self.cls_lambda = config.get("cls_lambda", self.cls_lambda)
        
        # T5 load first to set text model d_model
        self.t5 = T5ForConditionalGeneration.from_pretrained(p / "text_model")
        self.d_model = self.t5.config.d_model
        
        # Re-initialize visual backbone
        self._init_visual_encoder(self.visual_backbone)
        
        # Re-initialize projection layer
        self.proj = nn.Linear(self.num_visual_features, self.d_model)
        self.classifier = nn.Linear(self.num_visual_features, self.num_chexpert_labels)
        
        weights = torch.load(p / "vision_t5_weights.bin", map_location="cpu")
        self.proj.load_state_dict(weights["proj_state_dict"])
        if "classifier_state_dict" in weights:
            self.classifier.load_state_dict(weights["classifier_state_dict"])
        if weights["visual_features_state_dict"] is not None and not self.freeze_visual_encoder:
            self.visual_features.load_state_dict(weights["visual_features_state_dict"])
