import { access, readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import type { CatalogPlatform } from "./catalog/types";

export type { CatalogPlatform };

const execFileAsync = promisify(execFile);

export const PLATFORMS: CatalogPlatform[] = ["x5", "s", "x3"];

/** One platform distribution inside the model repository. */
export interface SourceEntry {
  /**
   * `worktree` reads the checked-out platform distribution; `tag` reads an
   * immutable annotated release tag through `git show`. Both modes share the
   * same manifest layout probe, so no platform is privileged.
   */
  mode: "worktree" | "tag";
  /** Platform root, repository-relative. Required by `worktree` mode. */
  path?: string;
  /** Annotated tag read by `tag` mode. */
  tag?: string;
  /** Path of the platform inside the tag tree. Legacy tags place it at the root. */
  tree_prefix?: string;
  /** Preferred manifest directory inside the platform root. */
  manifest_root?: string;
  /**
   * File that holds this platform's release version, relative to the platform
   * root. Worktree mode reads this exact path, strictly. The unified layout
   * colocates it with the manifests (`docs/release/<platform>/VERSION`)
   * because one repository root can no longer hold three disagreeing version
   * files. Defaults to `VERSION`. Tag sources ignore this field: a pinned
   * release is read with the layout that tag itself carries.
   */
  version_file?: string;
  /** Git ref used for repository source links (`blob/<ref>/...`). */
  link_ref: string;
  /** Path prefix of the platform inside `link_ref`. Empty for legacy layouts. */
  link_prefix: string;
}

export interface SourcesDocument {
  schema_version: number;
  repository: string;
  sources: Record<string, SourceEntry>;
}

export interface PlatformSource {
  platform: CatalogPlatform;
  kind: SourceEntry["mode"];
  /** Absolute platform root when read from the worktree. */
  worktreeRoot?: string;
  /** Git ref that holds the platform tree when read from a tag. */
  ref?: string;
  /** Path of the platform inside the ref tree; also the repository link prefix for that ref. */
  treePrefix: string;
  /** Manifest directory, relative to the platform root. */
  manifestDirectory: string;
  /**
   * File that holds this platform's release version, relative to the platform
   * root. For worktree sources this is the configured path; for tag sources it
   * is resolved from the layout the selected tag actually carries.
   */
  versionFile: string;
  /** Git ref used for repository source links. */
  linkRef: string;
  /** Path prefix of the platform inside `linkRef`; empty for legacy layouts. */
  linkPrefix: string;
}

/** A pinned platform reads an immutable tag instead of the checked-out tree. */
export interface SourcePin {
  tag: string;
  /** Path of the platform inside the tag tree. Defaults to the legacy root layout. */
  treePrefix?: string;
}

const MANIFEST_PROBE_ORDER = (preferred?: string): string[] => [
  ...(preferred ? [preferred] : []),
  "docs/release",
  "release"
].filter((directory, index, all) => all.indexOf(directory) === index);

async function gitShow(repositoryRoot: string, ref: string, path: string): Promise<string> {
  const result = await execFileAsync("git", ["-C", repositoryRoot, "show", `${ref}:${path}`], { encoding: "utf8" });
  return result.stdout;
}

/** Blob cache shared by every evidence read in this process. */
const blobCache = new Map<string, Promise<string>>();

/**
 * Reads one blob at an immutable ref from the repository that hosts the
 * platform histories. Evidence records cite `(ref, path)` precisely because
 * the worktree copy may since have been rewritten; this reads the revision the
 * record names. Refs outside this repository's object store (shallow clones,
 * pruned histories) fail like a missing file.
 */
export function readRepositoryBlob(repositoryRoot: string, ref: string, path: string): Promise<string> {
  const key = `${repositoryRoot}\0${ref}\0${path}`;
  let blob = blobCache.get(key);
  if (!blob) {
    blob = gitShow(repositoryRoot, ref, path);
    // Only fulfilled reads stay cached; a failure is retried by the next caller.
    blob.catch(() => blobCache.delete(key));
    blobCache.set(key, blob);
  }
  return blob;
}

/** True when the blob exists at the ref; never reads its content. */
export async function repositoryBlobExists(repositoryRoot: string, ref: string, path: string): Promise<boolean> {
  try {
    await execFileAsync("git", ["-C", repositoryRoot, "cat-file", "-e", `${ref}:${path}`]);
    return true;
  } catch {
    return false;
  }
}

/** Reads a file from a worktree source, or from the blob at a tag source's ref. */
export async function readSourceFile(repositoryRoot: string, source: PlatformSource, relativePath: string): Promise<string> {
  if (source.kind === "worktree") {
    return readFile(resolve(repositoryRoot, source.worktreeRoot!, relativePath), "utf8");
  }
  return gitShow(repositoryRoot, source.ref!, joinTreePath(source.treePrefix, relativePath));
}

export async function sourceFileExists(repositoryRoot: string, source: PlatformSource, relativePath: string): Promise<boolean> {
  if (source.kind === "worktree") {
    try {
      await access(resolve(repositoryRoot, source.worktreeRoot!, relativePath));
      return true;
    } catch {
      return false;
    }
  }
  try {
    await execFileAsync("git", ["-C", repositoryRoot, "cat-file", "-e", `${source.ref}:${joinTreePath(source.treePrefix, relativePath)}`]);
    return true;
  } catch {
    return false;
  }
}

function joinTreePath(prefix: string, relativePath: string): string {
  return [prefix.replace(/^\/+|\/+$/g, ""), relativePath.replace(/^\/+/g, "")].filter(Boolean).join("/");
}

/** Resolves the directory that holds this platform's release manifests. */
async function resolveManifestDirectory(repositoryRoot: string, source: PlatformSource): Promise<string> {
  const candidates = MANIFEST_PROBE_ORDER(source.manifestDirectory);
  for (const directory of candidates) {
    if (await sourceFileExists(repositoryRoot, source, `${directory}/models.yaml`)) return directory;
  }
  throw new Error(
    `${source.platform}: no models.yaml found under ${source.kind === "worktree" ? source.worktreeRoot : source.ref} `
    + `(tried ${candidates.map((candidate) => `${candidate}/`).join(", ")})`
  );
}

/**
 * Candidate VERSION paths inside one platform root, in probe order: the
 * unified layout colocates the version with the manifests; the legacy layouts
 * kept it at the platform root.
 */
const VERSION_PROBE_ORDER = (manifestDirectory: string): string[] => [
  `${manifestDirectory}/VERSION`,
  "VERSION"
].filter((candidate, index, all) => all.indexOf(candidate) === index);

/**
 * Resolves the VERSION path a source reads, from the manifest directory that
 * resolveManifestDirectory actually selected for it — pass that resolved
 * directory explicitly, not the source's original preferred field. A worktree
 * source reads exactly the configured file — the unified layout is fixed, so
 * a missing file must fail the build instead of silently falling back. A tag
 * source describes a historical checkout whose layout the pin selects, so the
 * worktree's configured path cannot stand in for it: the version is resolved
 * from the layout that tag actually carries, probed like the manifest
 * directory. A tag carrying manifests but no VERSION at all is a broken
 * release, not a default.
 */
async function resolveVersionFile(
  repositoryRoot: string,
  source: PlatformSource,
  manifestDirectory: string
): Promise<string> {
  if (source.kind === "worktree") return source.versionFile;
  const candidates = VERSION_PROBE_ORDER(manifestDirectory);
  for (const candidate of candidates) {
    if (await repositoryBlobExists(repositoryRoot, source.ref!, joinTreePath(source.treePrefix, candidate))) {
      return candidate;
    }
  }
  throw new Error(
    `${source.platform}: no VERSION found under ${source.ref} `
    + `(tried ${candidates.map((candidate) => joinTreePath(source.treePrefix, candidate)).join(", ")})`
  );
}

export interface ResolveSourcesOptions {
  repositoryRoot: string;
  sources: SourcesDocument;
  /** `platform -> tag` overrides that read a historical annotated tag instead. */
  pins?: Partial<Record<CatalogPlatform, SourcePin>>;
}

/**
 * Resolves every platform source symmetrically. The checked-out distribution is
 * the default for all three platforms; a pin replaces one platform with the
 * immutable tag it names, so a historical catalog version stays reproducible
 * without rewriting tags or changing published artifact URLs.
 */
export async function resolvePlatformSources(options: ResolveSourcesOptions): Promise<PlatformSource[]> {
  const pins = options.pins ?? {};
  const resolved: PlatformSource[] = [];
  for (const platform of PLATFORMS) {
    const entry = options.sources.sources[platform];
    if (!entry) throw new Error(`Catalog sources do not describe platform ${platform}`);
    const pin = pins[platform];

    let source: PlatformSource;
    if (pin) {
      source = {
        platform,
        kind: "tag",
        ref: pin.tag,
        treePrefix: pin.treePrefix ?? "",
        manifestDirectory: entry.manifest_root ?? "docs/release",
        versionFile: entry.version_file ?? "VERSION",
        // A pinned build keeps the layout frozen with the tag it names.
        linkRef: pin.tag,
        linkPrefix: pin.treePrefix ?? ""
      };
    } else if (entry.mode === "tag") {
      if (!entry.tag) throw new Error(`${platform}: tag mode requires a tag`);
      source = {
        platform,
        kind: "tag",
        ref: entry.tag,
        treePrefix: entry.tree_prefix ?? "",
        manifestDirectory: entry.manifest_root ?? "docs/release",
        versionFile: entry.version_file ?? "VERSION",
        linkRef: entry.tag,
        linkPrefix: entry.tree_prefix ?? ""
      };
    } else {
      if (!entry.path) throw new Error(`${platform}: worktree mode requires a path`);
      source = {
        platform,
        kind: "worktree",
        worktreeRoot: entry.path,
        treePrefix: entry.path,
        manifestDirectory: entry.manifest_root ?? "docs/release",
        versionFile: entry.version_file ?? "VERSION",
        linkRef: entry.link_ref,
        linkPrefix: entry.link_prefix
      };
    }

    if (source.kind === "tag") {
      const type = await execFileAsync("git", ["-C", options.repositoryRoot, "cat-file", "-t", source.ref!]);
      if (type.stdout.trim() !== "tag") throw new Error(`Annotated source tag required: ${source.ref}`);
    }
    // Resolve the manifest directory first, then VERSION against it: the
    // probe result is what the VERSION candidates must be colocated with, and
    // fields of one object literal cannot see each other's resolved values.
    const manifestDirectory = await resolveManifestDirectory(options.repositoryRoot, source);
    const versionFile = await resolveVersionFile(options.repositoryRoot, source, manifestDirectory);
    resolved.push({ ...source, manifestDirectory, versionFile });
  }
  return resolved;
}

export async function loadSourcesDocument(path: string): Promise<SourcesDocument> {
  const document = JSON.parse(await readFile(path, "utf8")) as SourcesDocument;
  if (document.schema_version !== 1) {
    throw new Error(`Unsupported catalog sources schema_version: ${document.schema_version}`);
  }
  return document;
}
