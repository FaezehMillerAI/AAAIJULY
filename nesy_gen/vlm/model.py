import torch
import torch.nn as nn
import torchvision
from typing import List
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from .chexpert_labels import (
    CHEXPERT_LABELS,
    labels_to_prompt_prefix,
    extract_chexpert_labels,
    get_edit_actions,
)


# ─────────────────────────────────────────────────────────────────────────────
# VisionT5 with Swin-T backbone + Diagnosis Classification Branch
# Architecture inspired by:
#   R2GenGPT  (Swin-T → frozen LLM, BLEU-1 0.491 IU-Xray)
#   PromptMRG (Diagnosis classification → token prompts, AAAI 2024)
#   KiUT      (knowledge-injected U-Transformer, CVPR 2023)
# ─────────────────────────────────────────────────────────────────────────────

class VisionT5(nn.Module):
    def __init__(
        self,
        text_model_name: str = "razent/SciFive-base-PMC",
        visual_backbone: str = "swin_tiny",
        freeze_visual_encoder: bool = False,
        use_diagnosis_prompts: bool = True,
        cls_lambda: float = 0.5,
    ):
        """
        Args:
            text_model_name:        HuggingFace T5 model identifier.
            visual_backbone:        One of 'swin_tiny', 'swin_base',
                                    'densenet121', 'resnet50',
                                    'efficientnet_b0', 'efficientnet_b4'.
            freeze_visual_encoder:  If True, visual encoder weights are frozen.
            use_diagnosis_prompts:  If True, prepend predicted CheXpert labels
                                    to the encoder prompt during generation.
            cls_lambda:             Weight of classification BCE loss relative
                                    to generation CE loss.
        """
        super().__init__()
        self.freeze_visual_encoder = freeze_visual_encoder
        self.visual_backbone = visual_backbone
        self.use_diagnosis_prompts = use_diagnosis_prompts
        self.cls_lambda = cls_lambda
        self.num_chexpert_labels = len(CHEXPERT_LABELS)

        # ── 1. Visual encoder ──────────────────────────────────────────────
        self._init_visual_encoder(visual_backbone)

        # ── 2. T5 text decoder ────────────────────────────────────────────
        self.t5 = T5ForConditionalGeneration.from_pretrained(text_model_name)
        self.d_model = self.t5.config.d_model

        # ── 3. Visual → T5 projection ─────────────────────────────────────
        self.proj = nn.Linear(self.num_visual_features, self.d_model)

        # ── 4. Diagnosis classification branch (PromptMRG-style) ──────────
        # Global-pooled visual features → CheXpert-14 logits
        self.classifier = nn.Linear(self.num_visual_features, self.num_chexpert_labels)
        self._cls_criterion = nn.BCEWithLogitsLoss()

    # ── Visual encoder initialisation ──────────────────────────────────────
    def _init_visual_encoder(self, visual_backbone: str):
        """Initialises the visual backbone and sets self.num_visual_features
        and self.is_swin (True if output is already (B, N, C))."""
        self.is_swin = False

        if visual_backbone == "swin_tiny":
            try:
                import timm
            except ImportError:
                raise ImportError("timm is required for Swin backbone: pip install timm")
            # global_pool='' → returns (B, H, W, C) spatial features
            self.visual_features = timm.create_model(
                "swin_tiny_patch4_window7_224",
                pretrained=True,
                num_classes=0,
                global_pool="",
            )
            self.num_visual_features = 768   # Swin-T last-stage channels
            self.is_swin = True

        elif visual_backbone == "swin_base":
            try:
                import timm
            except ImportError:
                raise ImportError("timm is required for Swin backbone: pip install timm")
            self.visual_features = timm.create_model(
                "swin_base_patch4_window7_224",
                pretrained=True,
                num_classes=0,
                global_pool="",
            )
            self.num_visual_features = 1024  # Swin-B last-stage channels
            self.is_swin = True

        elif visual_backbone == "densenet121":
            enc = torchvision.models.densenet121(
                weights=torchvision.models.DenseNet121_Weights.DEFAULT
            )
            self.visual_features = enc.features
            self.num_visual_features = 1024

        elif visual_backbone == "resnet50":
            enc = torchvision.models.resnet50(
                weights=torchvision.models.ResNet50_Weights.DEFAULT
            )
            self.visual_features = nn.Sequential(*list(enc.children())[:-2])
            self.num_visual_features = 2048

        elif visual_backbone == "efficientnet_b0":
            enc = torchvision.models.efficientnet_b0(
                weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT
            )
            self.visual_features = enc.features
            self.num_visual_features = 1280

        elif visual_backbone == "efficientnet_b4":
            enc = torchvision.models.efficientnet_b4(
                weights=torchvision.models.EfficientNet_B4_Weights.DEFAULT
            )
            self.visual_features = enc.features
            self.num_visual_features = 1792

        else:
            raise ValueError(f"Unsupported visual backbone: {visual_backbone!r}")

        if self.freeze_visual_encoder:
            for param in self.visual_features.parameters():
                param.requires_grad = False
        else:
            # Partial fine-tuning: Freeze early feature layers (patch_embed, stage 0 & 1)
            # to accelerate backpropagation, while keeping semantic stages (2 & 3) unfrozen.
            if hasattr(self.visual_features, "patch_embed"):
                for param in self.visual_features.patch_embed.parameters():
                    param.requires_grad = False
            if hasattr(self.visual_features, "layers"):
                # Swin has 4 layers (stages). Freeze the first 2.
                for i in range(min(2, len(self.visual_features.layers))):
                    for param in self.visual_features.layers[i].parameters():
                        param.requires_grad = False


    # ── Shared feature extraction ──────────────────────────────────────────
    def _extract_visual_features(self, images: torch.Tensor, frozen_ctx: bool = False):
        """
        Returns:
            spatial_feats: (B, N, C)  — sequence of patch tokens for T5
            pooled_feats:  (B, C)     — global-pooled features for classifier
        """
        if frozen_ctx or (self.freeze_visual_encoder and not self.training):
            ctx = torch.no_grad()
        else:
            from contextlib import nullcontext
            ctx = nullcontext()

        with ctx:
            if self.freeze_visual_encoder:
                self.visual_features.eval()
            raw = self.visual_features(images)

        # ── Swin returns (B, H, W, C) or (B, H*W, C) depending on timm ver ──
        if self.is_swin:
            if raw.dim() == 4:          # (B, H, W, C)
                B, H, W, C = raw.shape
                spatial_feats = raw.reshape(B, H * W, C)   # (B, 49, 768)
            else:                        # (B, N, C) already
                spatial_feats = raw
        else:
            # CNN: (B, C, H, W) → (B, N, C)
            spatial_feats = raw.flatten(2).transpose(1, 2)

        pooled_feats = spatial_feats.mean(dim=1)   # (B, C)
        return spatial_feats, pooled_feats

    # ── Forward (training) ────────────────────────────────────────────────
    def forward(
        self,
        images,
        decoder_input_ids=None,
        labels=None,
        encoder_input_ids=None,
        encoder_attention_mask=None,
        chexpert_labels=None,          # (B, 14) float tensor for classification
    ):
        spatial_feats, pooled_feats = self._extract_visual_features(images)

        batch_size = spatial_feats.size(0)
        device     = images.device

        # ── Classification branch ─────────────────────────────────────────
        cls_logits = torch.zeros((batch_size, self.num_chexpert_labels), device=device)
        cls_loss   = torch.tensor(0.0, device=device)
        if self.use_diagnosis_prompts:
            cls_logits = self.classifier(pooled_feats)   # (B, 14)
            if chexpert_labels is not None:
                cls_loss = self._cls_criterion(
                    cls_logits, chexpert_labels.float().to(device)
                )

        # ── Visual tokens → T5 embedding space ───────────────────────────
        visual_embeds = self.proj(spatial_feats)     # (B, N, d_model)
        vis_mask = torch.ones(
            (batch_size, visual_embeds.size(1)), dtype=torch.long, device=device
        )

        if encoder_input_ids is not None:
            # Embed the text tokens
            text_embeds = self.t5.encoder.embed_tokens(encoder_input_ids) # (B, seq_len, d_model)
            
            # Combine in the input embedding space
            combined_embeds = torch.cat([visual_embeds, text_embeds], dim=1) # (B, N + seq_len, d_model)
            combined_mask = (
                torch.cat([vis_mask, encoder_attention_mask], dim=1)
                if encoder_attention_mask is not None
                else vis_mask
            )
            
            # Pass combined embeds through the encoder so they fully interact
            encoder_outputs = self.t5.encoder(
                inputs_embeds=combined_embeds,
                attention_mask=combined_mask,
            )
        else:
            encoder_outputs = self.t5.encoder(
                inputs_embeds=visual_embeds,
                attention_mask=vis_mask,
            )
            combined_mask = vis_mask

        # If labels are provided but decoder_input_ids is None, shift right manually
        # to avoid passing labels directly to self.t5 (which forces standard CE loss)
        if decoder_input_ids is None and labels is not None:
            decoder_input_ids = self.t5._shift_right(labels)

        t5_out  = self.t5(
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            attention_mask=combined_mask,
        )

        logits = t5_out.logits
        gen_loss = torch.tensor(0.0, device=device)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=-100)
            gen_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.to(device).view(-1))

        # ── Combined loss: generation CE + λ × classification BCE ─────────
        total_loss = gen_loss + self.cls_lambda * cls_loss if self.use_diagnosis_prompts else gen_loss


        # Monkey-patch loss so trainer code stays unchanged
        t5_out.loss      = total_loss
        t5_out.gen_loss  = gen_loss
        t5_out.cls_loss  = cls_loss
        t5_out.cls_logits = cls_logits
        return t5_out

    # ── Generate (inference) ──────────────────────────────────────────────
    def generate(
        self,
        images,
        encoder_input_ids=None,
        encoder_attention_mask=None,
        tokenizer=None,
        template_reports: List[str] = None,  # list of template strings
        **kwargs,
    ):
        """
        At inference:
          1. Run visual encoder → classifier → predicted CheXpert labels
          2. If use_diagnosis_prompts and template_reports provided:
             Compute target edits and build prompt:
             "template: [text] edit: [actions]. generate report: [indication]"
          3. Else, prepend simple diagnosis prefix to prompt.
          4. Run T5 generate.
        """
        with torch.no_grad():
            spatial_feats, pooled_feats = self._extract_visual_features(images, frozen_ctx=True)

            batch_size = spatial_feats.size(0)
            device     = images.device

            # ── Step 1: predict diagnosis labels ─────────────────────────
            cls_logits = None
            cls_probs = None
            pred_labels = None
            if self.use_diagnosis_prompts:
                cls_logits  = self.classifier(pooled_feats)      # (B, 14)
                cls_probs   = torch.sigmoid(cls_logits)          # (B, 14)
                pred_labels = (cls_probs > 0.5).cpu().tolist()   # list[list[bool]]


            # ── Step 2: inject NeSy-CARE edit prompt or diagnosis prefix ──
            if self.use_diagnosis_prompts and tokenizer is not None and encoder_input_ids is not None:
                # Decode existing prompt tokens (e.g. "generate report: chest pain")
                raw_prompts = tokenizer.batch_decode(
                    encoder_input_ids, skip_special_tokens=True
                )
                
                aug_prompts = []
                for idx, (prompt_text, lbl_vec) in enumerate(zip(raw_prompts, pred_labels)):
                    template_report = (
                        template_reports[idx].strip()
                        if (template_reports is not None and idx < len(template_reports))
                        else ""
                    )
                    
                    if template_report:
                        tpl_floats = extract_chexpert_labels(template_report)
                        edit_actions = get_edit_actions(
                            tpl_floats,
                            [float(v) for v in lbl_vec],
                            cls_probs[idx].cpu().tolist()
                        )
                        aug = (
                            f"template: {template_report} "
                            f"edit: {edit_actions}. "
                            f"{prompt_text}"
                        )
                    else:
                        prefix = labels_to_prompt_prefix([float(v) for v in lbl_vec])
                        aug = f"{prefix} {prompt_text}".strip() if prefix else prompt_text
                        
                    aug_prompts.append(aug)

                # Re-encode augmented prompts
                enc = tokenizer(
                    aug_prompts,
                    max_length=self.t5.config.n_positions if hasattr(self.t5.config, "n_positions") else 512,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).to(device)
                encoder_input_ids    = enc["input_ids"]
                encoder_attention_mask = enc["attention_mask"]

            # ── Step 3: project visual tokens ────────────────────────────
            visual_embeds = self.proj(spatial_feats)        # (B, N, d_model)
            vis_mask = torch.ones(
                (batch_size, visual_embeds.size(1)), dtype=torch.long, device=device
            )

            if encoder_input_ids is not None:
                text_embeds = self.t5.encoder.embed_tokens(encoder_input_ids)
                combined_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
                combined_mask = (
                    torch.cat([vis_mask, encoder_attention_mask], dim=1)
                    if encoder_attention_mask is not None
                    else vis_mask
                )
                encoder_outputs = self.t5.encoder(
                    inputs_embeds=combined_embeds,
                    attention_mask=combined_mask,
                )
            else:
                encoder_outputs = self.t5.encoder(
                    inputs_embeds=visual_embeds,
                    attention_mask=vis_mask,
                )
                combined_mask = vis_mask

            return self.t5.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=combined_mask,
                **kwargs,
            )



    # ── Checkpoint ────────────────────────────────────────────────────────
    def save_checkpoint(self, path: str):
        import json
        from pathlib import Path
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "proj_state_dict":            self.proj.state_dict(),
                "classifier_state_dict":      self.classifier.state_dict(),
                "visual_features_state_dict": (
                    self.visual_features.state_dict()
                    if not self.freeze_visual_encoder
                    else None
                ),
            },
            p / "vision_t5_weights.bin",
        )
        self.t5.save_pretrained(p / "text_model")

        config = {
            "d_model":               self.d_model,
            "freeze_visual_encoder": self.freeze_visual_encoder,
            "num_visual_features":   self.num_visual_features,
            "visual_backbone":       self.visual_backbone,
            "use_diagnosis_prompts": self.use_diagnosis_prompts,
            "cls_lambda":            self.cls_lambda,
            "text_model_name": (
                self.t5.config._name_or_path
                if hasattr(self.t5.config, "_name_or_path")
                else "razent/SciFive-base-PMC"
            ),
        }
        for fname in ("config.json", "r2gen_t5_config.json"):
            with open(p / fname, "w") as f:
                json.dump(config, f, indent=2)

    def load_checkpoint(self, path: str):
        import json
        from pathlib import Path
        p = Path(path)

        cfg_file = p / "config.json" if (p / "config.json").exists() else p / "r2gen_t5_config.json"
        with open(cfg_file) as f:
            config = json.load(f)

        self.freeze_visual_encoder = config.get("freeze_visual_encoder", self.freeze_visual_encoder)
        self.visual_backbone       = config.get("visual_backbone", "swin_tiny")
        self.use_diagnosis_prompts = config.get("use_diagnosis_prompts", True)
        self.cls_lambda            = config.get("cls_lambda", 0.5)

        self.t5      = T5ForConditionalGeneration.from_pretrained(p / "text_model")
        self.d_model = self.t5.config.d_model

        self._init_visual_encoder(self.visual_backbone)
        self.proj       = nn.Linear(self.num_visual_features, self.d_model)
        self.classifier = nn.Linear(self.num_visual_features, self.num_chexpert_labels)

        weights = torch.load(p / "vision_t5_weights.bin", map_location="cpu")
        self.proj.load_state_dict(weights["proj_state_dict"])
        if "classifier_state_dict" in weights:
            self.classifier.load_state_dict(weights["classifier_state_dict"])
        if weights.get("visual_features_state_dict") is not None and not self.freeze_visual_encoder:
            self.visual_features.load_state_dict(weights["visual_features_state_dict"])
