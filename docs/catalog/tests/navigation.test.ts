import { describe, expect, it } from "vitest";
import { readCatalogQuery, writeCatalogQuery } from "../src/ui/navigation";
import { DEFAULT_QUERY } from "../src/ui/filters";

describe("directory URL state", () => {
  it("round trips directory filters independently of model hardware and task", () => {
    const query = { ...DEFAULT_QUERY, text: "YOLOv8", platform: "x5", tasks: ["object-detection"] };
    const url = writeCatalogQuery(new URL("https://example.test/?model=yolov8&hardware=s100&task=pose&view=cards"), query);
    expect(readCatalogQuery(url)).toEqual(query);
    expect(url.searchParams.get("hardware")).toBe("s100");
    expect(url.searchParams.get("task")).toBe("pose");
    expect(url.searchParams.get("view")).toBe("cards");
  });
  it("rejects unknown enum values without losing search", () => {
    const query = readCatalogQuery(new URL("https://example.test/?q=test&sort=bad&benchmark=bad&platform=RDK%20X5"));
    expect(query.sort).toBe("name");
    expect(query.benchmark).toBe("all");
    expect(query.platform).toBe("x5");
    expect(query.text).toBe("test");
  });
});
