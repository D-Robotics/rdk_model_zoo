import type { HardwareId, Locale } from "../catalog/types";

/** Shared inputs for the detail page and its focused view components. */
export interface DetailContext {
  locale: Locale;
  repositoryUrl: string;
  releaseTag: string;
  hardware?: HardwareId;
  task?: string;
  onSelectionChange?: (hardware: HardwareId, task: string) => void;
  /** Kept for links and integrations written before hardware tabs were added. */
  platform?: string;
}
