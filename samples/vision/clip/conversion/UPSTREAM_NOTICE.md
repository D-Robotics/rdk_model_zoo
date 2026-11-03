# OpenAI CLIP ViT-B/32

The independent export uses [OpenAI CLIP](https://github.com/openai/CLIP/tree/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6),
commit `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`.
Code and model weights retain the upstream [MIT license](licenses/LICENSE.clip).

Official checkpoint:
`https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt`.
Its complete publisher SHA-256 is
`40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`.
The BPE vocabulary SHA-256 is
`924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a`.
`export_pair.py` also checks the exact upstream model and tokenizer source bytes.

The image graph embeds the official CLIP channel normalization. Native input
remains RGB/NCHW float32 in [0,1], using the sample's OpenCV cubic short-side
resize and center crop. This deployment geometry is recorded separately from
the upstream Pillow transform. The text graph retains the learned text tower
and dynamic int32 [prompts,77] BPE input. Matching uses normalized features and
cosine scores without softmax. No text tower is replaced or represented as a
BPU model.

This does not establish the provenance of the historical default X5 pair.
