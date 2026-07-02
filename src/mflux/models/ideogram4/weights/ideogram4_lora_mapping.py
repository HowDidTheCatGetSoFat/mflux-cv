from mflux.models.common.lora.mapping.lora_mapping import LoRAMapping, LoRATarget


class Ideogram4LoRAMapping(LoRAMapping):
    @staticmethod
    def get_mapping() -> list[LoRATarget]:
        targets = [
            LoRATarget(
                model_path="input_proj",
                possible_up_patterns=[
                    "transformer.input_proj.lora_B.weight",
                    "diffusion_model.input_proj.lora_B.weight",
                ],
                possible_down_patterns=[
                    "transformer.input_proj.lora_A.weight",
                    "diffusion_model.input_proj.lora_A.weight",
                ],
                possible_alpha_patterns=[
                    "transformer.input_proj.alpha",
                    "diffusion_model.input_proj.alpha",
                ],
            ),
            LoRATarget(
                model_path="llm_cond_proj",
                possible_up_patterns=[
                    "transformer.llm_cond_proj.lora_B.weight",
                    "diffusion_model.llm_cond_proj.lora_B.weight",
                ],
                possible_down_patterns=[
                    "transformer.llm_cond_proj.lora_A.weight",
                    "diffusion_model.llm_cond_proj.lora_A.weight",
                ],
                possible_alpha_patterns=[
                    "transformer.llm_cond_proj.alpha",
                    "diffusion_model.llm_cond_proj.alpha",
                ],
            ),
            LoRATarget(
                model_path="adaln_proj",
                possible_up_patterns=[
                    "transformer.adaln_proj.lora_B.weight",
                    "diffusion_model.adaln_proj.lora_B.weight",
                ],
                possible_down_patterns=[
                    "transformer.adaln_proj.lora_A.weight",
                    "diffusion_model.adaln_proj.lora_A.weight",
                ],
                possible_alpha_patterns=[
                    "transformer.adaln_proj.alpha",
                    "diffusion_model.adaln_proj.alpha",
                ],
            ),
            LoRATarget(
                model_path="final_layer.linear",
                possible_up_patterns=[
                    "transformer.final_layer.linear.lora_B.weight",
                    "diffusion_model.final_layer.linear.lora_B.weight",
                    "lora_unet_final_linear.lora_up.weight",
                ],
                possible_down_patterns=[
                    "transformer.final_layer.linear.lora_A.weight",
                    "diffusion_model.final_layer.linear.lora_A.weight",
                    "lora_unet_final_linear.lora_down.weight",
                ],
                possible_alpha_patterns=[
                    "transformer.final_layer.linear.alpha",
                    "diffusion_model.final_layer.linear.alpha",
                    "lora_unet_final_linear.alpha",
                ],
            ),
            LoRATarget(
                model_path="layers.{block}.adaln_modulation",
                possible_up_patterns=[
                    "transformer.layers.{block}.adaln_modulation.lora_B.weight",
                    "diffusion_model.layers.{block}.adaln_modulation.lora_B.weight",
                    "lora_unet_layers_{block}_adaln_modulation.lora_up.weight",
                ],
                possible_down_patterns=[
                    "transformer.layers.{block}.adaln_modulation.lora_A.weight",
                    "diffusion_model.layers.{block}.adaln_modulation.lora_A.weight",
                    "lora_unet_layers_{block}_adaln_modulation.lora_down.weight",
                ],
                possible_alpha_patterns=[
                    "transformer.layers.{block}.adaln_modulation.alpha",
                    "diffusion_model.layers.{block}.adaln_modulation.alpha",
                    "lora_unet_layers_{block}_adaln_modulation.alpha",
                ],
            ),
            LoRATarget(
                model_path="layers.{block}.attention.qkv",
                possible_up_patterns=[
                    "transformer.layers.{block}.attention.qkv.lora_B.weight",
                    "diffusion_model.layers.{block}.attention.qkv.lora_B.weight",
                ],
                possible_down_patterns=[
                    "transformer.layers.{block}.attention.qkv.lora_A.weight",
                    "diffusion_model.layers.{block}.attention.qkv.lora_A.weight",
                ],
                possible_alpha_patterns=[
                    "transformer.layers.{block}.attention.qkv.alpha",
                    "diffusion_model.layers.{block}.attention.qkv.alpha",
                ],
            ),
            LoRATarget(
                model_path="layers.{block}.attention.o",
                possible_up_patterns=[
                    "transformer.layers.{block}.attention.o.lora_B.weight",
                    "diffusion_model.layers.{block}.attention.o.lora_B.weight",
                ],
                possible_down_patterns=[
                    "transformer.layers.{block}.attention.o.lora_A.weight",
                    "diffusion_model.layers.{block}.attention.o.lora_A.weight",
                ],
                possible_alpha_patterns=[
                    "transformer.layers.{block}.attention.o.alpha",
                    "diffusion_model.layers.{block}.attention.o.alpha",
                ],
            ),
            LoRATarget(
                model_path="layers.{block}.feed_forward.w1",
                possible_up_patterns=[
                    "transformer.layers.{block}.feed_forward.w1.lora_B.weight",
                    "diffusion_model.layers.{block}.feed_forward.w1.lora_B.weight",
                ],
                possible_down_patterns=[
                    "transformer.layers.{block}.feed_forward.w1.lora_A.weight",
                    "diffusion_model.layers.{block}.feed_forward.w1.lora_A.weight",
                ],
                possible_alpha_patterns=[
                    "transformer.layers.{block}.feed_forward.w1.alpha",
                    "diffusion_model.layers.{block}.feed_forward.w1.alpha",
                ],
            ),
            LoRATarget(
                model_path="layers.{block}.feed_forward.w2",
                possible_up_patterns=[
                    "transformer.layers.{block}.feed_forward.w2.lora_B.weight",
                    "diffusion_model.layers.{block}.feed_forward.w2.lora_B.weight",
                ],
                possible_down_patterns=[
                    "transformer.layers.{block}.feed_forward.w2.lora_A.weight",
                    "diffusion_model.layers.{block}.feed_forward.w2.lora_A.weight",
                ],
                possible_alpha_patterns=[
                    "transformer.layers.{block}.feed_forward.w2.alpha",
                    "diffusion_model.layers.{block}.feed_forward.w2.alpha",
                ],
            ),
            LoRATarget(
                model_path="layers.{block}.feed_forward.w3",
                possible_up_patterns=[
                    "transformer.layers.{block}.feed_forward.w3.lora_B.weight",
                    "diffusion_model.layers.{block}.feed_forward.w3.lora_B.weight",
                ],
                possible_down_patterns=[
                    "transformer.layers.{block}.feed_forward.w3.lora_A.weight",
                    "diffusion_model.layers.{block}.feed_forward.w3.lora_A.weight",
                ],
                possible_alpha_patterns=[
                    "transformer.layers.{block}.feed_forward.w3.alpha",
                    "diffusion_model.layers.{block}.feed_forward.w3.alpha",
                ],
            ),
        ]
        # DoRA: derive dora_scale key patterns from each target's LoRA-A (down) patterns
        # ("<base>.lora_A.weight" -> "<base>.dora_scale") so DoRA adapters load their magnitude.
        _suffixes = (".lora_A.weight", ".lora_A.default.weight", ".lora_down.weight", ".lora_down.default.weight")
        for target in targets:
            dora_patterns: list[str] = []
            for down in target.possible_down_patterns:
                for suffix in _suffixes:
                    if down.endswith(suffix):
                        dora_patterns.append(down[: -len(suffix)] + ".dora_scale")
                        break
            target.possible_dora_scale_patterns = dora_patterns
        return targets
