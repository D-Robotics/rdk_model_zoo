"""Register Gemma4 model builders with the Leap LLM model factory.

Copy this extension block into Leap LLM's ``model_factory.py`` as described by
the conversion documentation.
"""

@register_model("gemma4-e2b-vision", ["nash-e", "nash-m", "nash-p"])
def _build_gemma4_e2b_vision(args):
    from leap_llm.apis.model.gemma4 import Gemma4VisionApi

    return Gemma4VisionApi(
        input_model_path=args.input_model_path,
        output_model_path=args.output_model_path,
        calib_image_path=args.calib_image_path,
        device=args.device,
        model_type="gemma4-e2b",
        core_num=args.vit_core_num,
    )


@register_model("gemma4-e2b-text", ["nash-e", "nash-m", "nash-p"])
def _build_gemma4_e2b_text(args):
    from leap_llm.apis.model.gemma4 import Gemma4TextApi

    return Gemma4TextApi(
        input_model_path=args.input_model_path,
        output_model_path=args.output_model_path,
        calib_text_path=args.calib_text_path,
        chunk_size=args.chunk_size,
        cache_len=args.cache_len,
        device=args.device,
        model_type="gemma4-e2b",
        prefill_core_num=args.prefill_core_num,
        decode_core_num=args.decode_core_num,
    )


@register_model("gemma4-e4b-vision", ["nash-e", "nash-m", "nash-p"])
def _build_gemma4_e4b_vision(args):
    from leap_llm.apis.model.gemma4 import Gemma4VisionApi

    return Gemma4VisionApi(
        input_model_path=args.input_model_path,
        output_model_path=args.output_model_path,
        calib_image_path=args.calib_image_path,
        device=args.device,
        model_type="gemma4-e4b",
        core_num=args.vit_core_num,
    )


@register_model("gemma4-e4b-text", ["nash-e", "nash-m", "nash-p"])
def _build_gemma4_e4b_text(args):
    from leap_llm.apis.model.gemma4 import Gemma4TextApi

    return Gemma4TextApi(
        input_model_path=args.input_model_path,
        output_model_path=args.output_model_path,
        calib_text_path=args.calib_text_path,
        chunk_size=args.chunk_size,
        cache_len=args.cache_len,
        device=args.device,
        model_type="gemma4-e4b",
        prefill_core_num=args.prefill_core_num,
        decode_core_num=args.decode_core_num,
    )
