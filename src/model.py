
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    CLIPVisionModel,
    GPT2LMHeadModel,
    GPT2Tokenizer
)


class MiniMolmo(nn.Module):
    """Educational vision-language prototype inspired by Molmo."""

    def __init__(self, prefix_length=8):
        super().__init__()

        self.vision = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

        self.decoder = GPT2LMHeadModel.from_pretrained("gpt2")

        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.decoder.config.pad_token_id = self.tokenizer.pad_token_id

        self.prefix_length = prefix_length

        vd = self.vision.config.hidden_size
        dd = self.decoder.config.n_embd

        # Visual connector
        self.connector = nn.Sequential(
            nn.LayerNorm(vd),
            nn.Linear(vd, dd),
            nn.GELU(),
            nn.Linear(dd, dd)
        )

        # Question-to-vision projection
        self.question_to_vision = nn.Linear(dd, vd)

        # Question-guided visual attention
        self.visual_attention = nn.MultiheadAttention(
            embed_dim=vd,
            num_heads=8,
            batch_first=True
        )

        # Predict [x_min, y_min, x_max, y_max]
        self.bbox_head = nn.Sequential(
            nn.LayerNorm(vd * 2),
            nn.Linear(vd * 2, 512),
            nn.GELU(),
            nn.Linear(512, 4),
            nn.Sigmoid()
        )

        # Freeze pretrained models
        for p in self.vision.parameters():
            p.requires_grad = False

        for p in self.decoder.parameters():
            p.requires_grad = False

    def visual_features(self, pixels):
        with torch.no_grad():
            output = self.vision(pixel_values=pixels)

        # Exclude CLS token
        return output.last_hidden_state[:, 1:, :]

    def prefix(self, patches):
        pooled = F.adaptive_avg_pool1d(
            patches.transpose(1, 2),
            self.prefix_length
        ).transpose(1, 2)

        return self.connector(pooled)

    def question_features(self, prompts, device):
        tokens = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=48
        ).to(device)

        with torch.no_grad():
            embeddings = self.decoder.transformer.wte(
                tokens.input_ids
            )

        mask = tokens.attention_mask.unsqueeze(-1)

        features = (
            (embeddings * mask).sum(dim=1)
            / mask.sum(dim=1).clamp(min=1)
        )

        return features

    def predict_boxes(self, patches, prompts):
        question = self.question_features(
            prompts,
            patches.device
        )

        question_query = self.question_to_vision(question)

        query = question_query.unsqueeze(1)

        attended_visual, _ = self.visual_attention(
            query=query,
            key=patches,
            value=patches,
            need_weights=False
        )

        attended_visual = attended_visual.squeeze(1)

        combined = torch.cat(
            [attended_visual, question_query],
            dim=1
        )

        raw_boxes = self.bbox_head(combined)

        # Ensure x_min <= x_max and y_min <= y_max
        x1 = torch.minimum(raw_boxes[:, 0], raw_boxes[:, 2])
        y1 = torch.minimum(raw_boxes[:, 1], raw_boxes[:, 3])
        x2 = torch.maximum(raw_boxes[:, 0], raw_boxes[:, 2])
        y2 = torch.maximum(raw_boxes[:, 1], raw_boxes[:, 3])

        return torch.stack([x1, y1, x2, y2], dim=1)

    def training_loss(
        self,
        pixels,
        prompts,
        answers,
        boxes,
        box_valid,
        max_length=48
    ):
        patches = self.visual_features(pixels)

        pref = self.prefix(patches)

        pred_boxes = self.predict_boxes(patches, prompts)

        pids = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).input_ids.to(pixels.device)

        aids = self.tokenizer(
            answers,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).input_ids.to(pixels.device)

        p_emb = self.decoder.transformer.wte(pids)
        a_emb = self.decoder.transformer.wte(aids)

        embeds = torch.cat(
            [pref, p_emb, a_emb],
            dim=1
        )

        ignored_pref = torch.full(
            (pixels.size(0), self.prefix_length),
            -100,
            device=pixels.device,
            dtype=torch.long
        )

        ignored_prompt = torch.full_like(pids, -100)

        labels = aids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        labels = torch.cat(
            [ignored_pref, ignored_prompt, labels],
            dim=1
        )

        out = self.decoder(
            inputs_embeds=embeds,
            labels=labels,
            attention_mask=torch.ones(
                embeds.shape[:2],
                device=pixels.device,
                dtype=torch.long
            )
        )

        valid = box_valid.bool()

        if valid.any():
            bbox_loss = F.smooth_l1_loss(
                pred_boxes[valid],
                boxes[valid]
            )
        else:
            bbox_loss = pred_boxes.sum() * 0.0

        total_loss = out.loss + 2.0 * bbox_loss

        return (
            total_loss,
            out.loss.detach(),
            bbox_loss.detach()
        )

    @torch.inference_mode()
    def generate(self, pixels, prompt, max_new_tokens=40):
        self.eval()

        patches = self.visual_features(pixels)
        pref = self.prefix(patches)

        ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=48
        ).input_ids.to(pixels.device)

        prompt_embeds = self.decoder.transformer.wte(ids)

        embeds = torch.cat(
            [pref, prompt_embeds],
            dim=1
        )

        mask = torch.ones(
            embeds.shape[:2],
            device=pixels.device,
            dtype=torch.long
        )

        output = self.decoder.generate(
            inputs_embeds=embeds,
            attention_mask=mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            use_cache=True
        )

        text = self.tokenizer.decode(
            output[0],
            skip_special_tokens=True
        ).strip()

        predicted_box = self.predict_boxes(
            patches,
            [prompt]
        )[0].cpu().tolist()

        return text, predicted_box
