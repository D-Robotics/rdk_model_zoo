const archiveHost = 'archive.d-robotics.cc';
const llmHost = 'd-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com';

export function isPublishedDownloadSource(value) {
  try {
    const url = new URL(value);
    if (url.protocol !== 'https:' || url.username || url.password || url.port || url.search || url.hash) return false;
    if (url.hostname === archiveHost) {
      return /^\/downloads\/rdk_model_zoo_web\/rdk_(?:x5|s100|s100p|s600)\/[^/]+\/.+/.test(url.pathname);
    }
    return url.hostname === llmHost && url.pathname.startsWith('/llm/');
  } catch {
    return false;
  }
}
