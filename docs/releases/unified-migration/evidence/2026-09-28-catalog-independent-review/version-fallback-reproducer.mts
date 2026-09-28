import { loadSourcesDocument, resolvePlatformSources } from "../../../../../tools/catalog-publisher/src/sources.ts";
import { createTaggedRepository, fixtureManifests } from "../../../../../tools/catalog-publisher/tests/helpers/tag-repository.ts";
const sources = await loadSourcesDocument("tools/catalog-publisher/sources.json");
for (const rootVersion of [false, true]) {
 const files = { ...fixtureManifests("x5", "x5-v9.9.9", "1.0.0", "release"), "release/VERSION": "1.0.0\n", ...(rootVersion ? {"VERSION": "0.0.1\n"} : {}) };
 const root = await createTaggedRepository("x5-v9.9.9", files);
 try {
  const resolved = await resolvePlatformSources({ repositoryRoot: root, sources, pins: { x5: {tag:"x5-v9.9.9"} } });
  const x5 = resolved.find(s => s.platform === "x5");
  console.log(JSON.stringify({rootVersion, manifestDirectory:x5?.manifestDirectory, versionFile:x5?.versionFile, correct:x5?.versionFile === "release/VERSION"}));
 } catch(e) {console.log(JSON.stringify({rootVersion,error:String(e),correct:false}));}
}
