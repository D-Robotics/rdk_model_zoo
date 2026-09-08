"""MiniCPM5 Llama adapter for OpenExplorer LLM 2.0.0-beta1."""
from transformers import AutoConfig

from llm_compression.models.qwen2.qwen2_model import Qwen2
from llm_compression.registry_factory import MODEL_REGISTRY


@MODEL_REGISTRY
class MiniCPM5(Qwen2):
    """Adapt bias-free MiniCPM5 Llama weights to the SDK Qwen2 operators."""
    def build_model(self, model_dir):
        """Validate Llama configuration and preserve its original RoPE theta.

        Args:
            model_dir: Local directory containing the original HF checkpoint.

        Returns:
            SDK model constructed with the validated Llama configuration.
        """
        config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
        if config.model_type != 'llama':
            raise ValueError('MiniCPM5 adapter requires a Llama checkpoint')
        rope = getattr(config, 'rope_parameters', None) or config.rope_scaling or {}
        if rope.get('rope_type', rope.get('type', 'default')) != 'default':
            raise ValueError('Scaled RoPE is not validated by this adapter')
        if config.attention_bias or getattr(config, 'mlp_bias', False):
            raise ValueError('MiniCPM5 requires bias-free attention and MLP')
        # Transformers 5 moves rope_theta into rope_parameters. The SDK still
        # reads the legacy top-level attribute and otherwise defaults to 1e6.
        self.custom_config.model.text_config.rope_theta = rope.get(
            'rope_theta', getattr(config, 'rope_theta', 10000.0))
        return super().build_model(model_dir)

    def input_preprocess(self, message):
        """Tokenize a chat conversation once using the official template.

        Args:
            message: Ordered role/content dictionaries forming a conversation.

        Returns:
            Tokenizer batch containing input IDs and attention masks as tensors.
        """
        # Tokenize once: re-tokenizing template text can add a duplicate BOS.
        thinking = getattr(self.custom_config.model, 'enable_thinking', False)
        return self.tokenizer.apply_chat_template(
            message, tokenize=True, add_generation_prompt=True,
            enable_thinking=thinking, return_dict=True, return_tensors='pt')
