import { mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";

export function writeJson(path: string, obj: unknown) {
  mkdirSync(dirname(path), { recursive: true });
  const json = JSON.stringify(obj, null, 2);
  writeFileSync(path, json + "\n", { encoding: "utf8" });
}

export function writeText(path: string, text: string) {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, text + "\n", { encoding: "utf8" });
}
