import type { Locale } from "../catalog/types";
import { detailLabel, taskLabel } from "./detail-labels";

export interface TaskSelectorOptions {
  locale: Locale;
  modelId: string;
  tasks: string[];
  selectedTask: string;
  onSelect: (task: string) => void;
}

export interface TaskSelectorView {
  element: HTMLDivElement;
  setTasks(tasks: string[], selectedTask: string): void;
  destroy(): void;
}

export function createTaskSelector(options: TaskSelectorOptions): TaskSelectorView {
  const element = document.createElement("div");
  element.className = "model-detail-task-control";
  const label = document.createElement("label");
  label.htmlFor = `model-detail-task-${options.modelId}`;
  label.textContent = detailLabel(options.locale, "task");
  const select = document.createElement("select");
  select.id = label.htmlFor;
  select.dataset.control = "task";
  element.append(label, select);

  const onChange = (): void => options.onSelect(select.value);
  select.addEventListener("change", onChange);

  const setTasks = (tasks: string[], selectedTask: string): void => {
    select.replaceChildren();
    for (const task of tasks) {
      const option = document.createElement("option");
      option.value = task;
      option.textContent = taskLabel(options.locale, task);
      option.selected = task === selectedTask;
      select.append(option);
    }
    select.hidden = tasks.length <= 1;
    label.hidden = tasks.length <= 1;
    element.dataset.taskCount = String(tasks.length);
  };
  setTasks(options.tasks, options.selectedTask);

  return {
    element,
    setTasks,
    destroy(): void {
      select.removeEventListener("change", onChange);
    }
  };
}
