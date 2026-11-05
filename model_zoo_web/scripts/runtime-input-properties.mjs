// Preserve published physical-input facts when rendering converted model pages.
export function runtimeInputProperties(release, input) {
  const artifact = release.artifact || {};
  const converted = release.status === 'converted';
  const declared = converted ? release.conversion?.runtime_input || {} : {};
  const format = declared.format ? String(declared.format).toLowerCase() : artifact.runtime_input;
  return {
    format,
    resolution: declared.layout === 'NCTHW' && declared.shape?.length === 5
      ? declared.shape.slice(2) : [input.height, input.width],
    ...(converted && (declared.dtype || artifact.runtime_dtype) ? {dtype: declared.dtype || artifact.runtime_dtype} : {}),
    ...(converted && (declared.layout || artifact.runtime_layout) ? {payload_layout: declared.layout || artifact.runtime_layout} : {}),
    ...(artifact.runtime_tensor || {}),
  };
}
