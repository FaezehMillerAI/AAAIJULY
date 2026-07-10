import torch
import torch.nn as nn
import torchvision
from transformers import T5ForConditionalGeneration, T5Config
from transformers.modeling_outputs import BaseModelOutput

class VisionT5(nn.Module):
    def __init__(self, text_model_name: str = "t5-small", visual_backbone: str = "densenet121", freeze_visual_encoder: bool = True):
        super().__init__()
        self.freeze_visual_encoder = freeze_visual_encoder
        self.visual_backbone = visual_backbone
        
        # Initialize visual backbone features
        self._init_visual_encoder(visual_backbone)
                
        # Load T5 model
        self.t5 = T5ForConditionalGeneration.from_pretrained(text_model_name)
        self.d_model = self.t5.config.d_model
        
        # Projection layer: maps visual features to T5 embedding space (d_model)
        self.proj = nn.Linear(self.num_visual_features, self.d_model)

    def _init_visual_encoder(self, visual_backbone: str):
        if visual_backbone == "densenet121":
            self.visual_encoder = torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.DEFAULT)
            self.visual_features = self.visual_encoder.features
            self.num_visual_features = 1024
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
        else:
            raise ValueError(f"Unsupported visual backbone: {visual_backbone}")
            
        if self.freeze_visual_encoder:
            for param in self.visual_features.parameters():
                param.requires_grad = False
        
    def forward(self, images, decoder_input_ids=None, labels=None, encoder_input_ids=None, encoder_attention_mask=None):
        # Extract visual features
        if self.freeze_visual_encoder:
            with torch.no_grad():
                # We put visual features in eval mode if frozen
                self.visual_features.eval()
                visual_feats = self.visual_features(images)  # Shape: (batch, 1024, 7, 7)
        else:
            visual_feats = self.visual_features(images)
            
        batch_size = visual_feats.size(0)
        
        # Reshape to sequence of visual tokens: (batch, num_tokens, num_visual_features)
        # 7x7 = 49 visual tokens
        visual_feats = visual_feats.flatten(2).transpose(1, 2)  # (batch, 49, 1024)
        visual_tokens = self.proj(visual_feats)  # (batch, 49, d_model)
        
        # If encoder text input is provided, encode it and concatenate
        if encoder_input_ids is not None:
            text_encoder_outputs = self.t5.encoder(input_ids=encoder_input_ids, attention_mask=encoder_attention_mask)
            text_feats = text_encoder_outputs.last_hidden_state  # (batch, seq_len, d_model)
            
            # Combine visual tokens and text tokens
            combined_tokens = torch.cat([visual_tokens, text_feats], dim=1)  # (batch, 49 + seq_len, d_model)
            
            # Combine attention masks
            vis_attention_mask = torch.ones((batch_size, visual_tokens.size(1)), dtype=torch.long, device=images.device)
            if encoder_attention_mask is not None:
                combined_attention_mask = torch.cat([vis_attention_mask, encoder_attention_mask], dim=1)
            else:
                combined_attention_mask = vis_attention_mask
        else:
            combined_tokens = visual_tokens
            combined_attention_mask = torch.ones((batch_size, visual_tokens.size(1)), dtype=torch.long, device=images.device)
            
        # Create BaseModelOutput for encoder_outputs
        encoder_outputs = BaseModelOutput(last_hidden_state=combined_tokens)
        
        outputs = self.t5(
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            attention_mask=combined_attention_mask,
            labels=labels
        )
        return outputs

    def generate(self, images, encoder_input_ids=None, encoder_attention_mask=None, **kwargs):
        # Extract visual features
        with torch.no_grad():
            self.visual_features.eval()
            visual_feats = self.visual_features(images)
            visual_feats = visual_feats.flatten(2).transpose(1, 2)
            visual_tokens = self.proj(visual_feats)
            
            batch_size = visual_feats.size(0)
            if encoder_input_ids is not None:
                text_encoder_outputs = self.t5.encoder(input_ids=encoder_input_ids, attention_mask=encoder_attention_mask)
                text_feats = text_encoder_outputs.last_hidden_state
                combined_tokens = torch.cat([visual_tokens, text_feats], dim=1)
                vis_attention_mask = torch.ones((batch_size, visual_tokens.size(1)), dtype=torch.long, device=images.device)
                if encoder_attention_mask is not None:
                    combined_attention_mask = torch.cat([vis_attention_mask, encoder_attention_mask], dim=1)
                else:
                    combined_attention_mask = vis_attention_mask
            else:
                combined_tokens = visual_tokens
                combined_attention_mask = torch.ones((batch_size, visual_tokens.size(1)), dtype=torch.long, device=images.device)
                
            encoder_outputs = BaseModelOutput(last_hidden_state=combined_tokens)
            
            return self.t5.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=combined_attention_mask,
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
            "text_model_name": self.t5.config._name_or_path if hasattr(self.t5.config, "_name_or_path") else "razent/SciFive-base-PMC"
        }
        with open(p / "config.json", "w") as f:
            json.dump(config, f)
        # Keep old name for backward compat
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
        
        # T5 load first to set text model d_model
        self.t5 = T5ForConditionalGeneration.from_pretrained(p / "text_model")
        self.d_model = self.t5.config.d_model
        
        # Re-initialize visual backbone
        self._init_visual_encoder(self.visual_backbone)
        
        # Re-initialize projection layer
        self.proj = nn.Linear(self.num_visual_features, self.d_model)
        
        weights = torch.load(p / "vision_t5_weights.bin", map_location="cpu")
        self.proj.load_state_dict(weights["proj_state_dict"])
        if weights["visual_features_state_dict"] is not None and not self.freeze_visual_encoder:
            self.visual_features.load_state_dict(weights["visual_features_state_dict"])
