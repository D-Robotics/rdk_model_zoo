import type { Locale } from "../catalog/types";

/** Presentation-only shell: model records and filtering remain owned by the catalog. */
export function renderEditorialIntro(locale: Locale): HTMLElement {
  const zh = locale === "zh";
  const hero = document.createElement("section");
  hero.className = "catalog-hero";
  hero.setAttribute("aria-labelledby", "catalog-hero-title");
  const eyebrow = document.createElement("p");
  eyebrow.className = "editorial-eyebrow";
  eyebrow.textContent = "D-ROBOTICS / MODEL ZOO";
  const heading = document.createElement("h1");
  heading.id = "catalog-hero-title";
  heading.append(zh ? "找到模型，" : "Find your model.", document.createElement("br"),
    zh ? "让想法运行。" : "Make it run.");
  const description = document.createElement("p");
  description.className = "catalog-hero-description";
  description.textContent = zh
    ? "为你的 RDK 硬件选择模型。对比性能与精度，下载可直接运行的量化模型。"
    : "Find the right model for your RDK hardware. Compare performance and accuracy, then download a model ready to run.";
  const explore = document.createElement("a");
  explore.className = "editorial-primary";
  explore.href = "#model-directory";
  explore.append(zh ? "探索模型" : "Explore models");
  const arrow = document.createElement("span");
  arrow.setAttribute("aria-hidden", "true");
  arrow.textContent = "↓";
  explore.append(arrow);
  hero.append(eyebrow, heading, description, explore);
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
