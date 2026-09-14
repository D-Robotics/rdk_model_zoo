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
        linkRef: entry.link_ref,
        linkPrefix: entry.link_prefix
      };
    }

    if (source.kind === "tag") {
      const type = await execFileAsync("git", ["-C", options.repositoryRoot, "cat-file", "-t", source.ref!]);
      if (type.stdout.trim() !== "tag") throw new Error(`Annotated source tag required: ${source.ref}`);
    }
    resolved.push({ ...source, manifestDirectory: await resolveManifestDirectory(options.repositoryRoot, source) });
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
