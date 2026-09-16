import type { Catalog, Locale } from "../catalog/types";
import { HARDWARE_IDS } from "../catalog/variants";

/** Counts represent catalog coverage, not a claim of measured performance. */
export function renderEditorialIntro(locale: Locale, catalog?: Catalog): HTMLElement {
  const zh = locale === "zh";
  const hero = document.createElement("section");
  hero.className = "catalog-dashboard";
  hero.setAttribute("aria-labelledby", "catalog-hero-title");
  const heading = document.createElement("h1");
  heading.id = "catalog-hero-title";
  heading.textContent = zh ? "模型概览" : "Model overview";
  const counts = document.createElement("div");
  counts.className = "catalog-total-counts";
  const models = catalog?.models ?? [];
  const variants = models.flatMap(model => model.variants ?? []);
  for (const [key, count, label] of [
    ["families", models.length, zh ? "模型系列" : "Model families"],
    ["models", variants.length, zh ? "具体模型" : "Models by platform"]
  ] as const) {
    const item = document.createElement("div");
    const value = document.createElement("strong");
    value.dataset.count = key;
    value.textContent = count.toLocaleString(locale);
    const title = document.createElement("span");
    title.textContent = label;
    item.append(value, title);
    counts.append(item);
  }
  const note = document.createElement("p");
  note.className = "catalog-count-note";
  note.textContent = zh ? "具体模型按平台分别计数，包含不同输入尺寸及任务配置；收录不代表均已完成实测。" : "Models are counted per platform, input size and task. Catalog coverage does not imply every configuration has measured benchmarks.";
  const platforms = document.createElement("div");
  platforms.className = "catalog-platform-counts";
  for (const hardware of HARDWARE_IDS) {
    const link = document.createElement("a");
    link.dataset.platform = hardware;
    link.href = "?platform=" + hardware + "#model-directory";
    const name = document.createElement("strong");
    name.textContent = hardware.toUpperCase();
    const familyCount = models.filter(model => model.variants?.some(v => v.hardware === hardware)).length;
    const modelCount = variants.filter(v => v.hardware === hardware).length;
    const value = document.createElement("span");
    value.textContent = zh ? familyCount + " 个系列 · " + modelCount + " 个模型" : familyCount + " families · " + modelCount + " models";
    link.append(name, value);
    platforms.append(link);
  }
  hero.append(heading, counts, note, platforms);
  return hero;
}

export function renderDirectoryHeading(locale: Locale): HTMLElement {
  const section = document.createElement("div");
  section.className = "directory-heading";
  section.id = "model-directory";
  section.tabIndex = -1;
  const heading = document.createElement("h2");
  heading.textContent = locale === "zh" ? "模型目录" : "The model collection";
  const hint = document.createElement("p");
  hint.textContent = locale === "zh" ? "选择硬件，找到适合你的模型。" : "Your hardware. Your next model.";
  section.append(heading, hint);
  return section;
}
