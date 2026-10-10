/**
 * Files translated with their structure kept. The port of `ynlibs.translation.files` and
 * `ynlibs.translation.export`.
 */

import { exportString, mapCues, splitFrontMatter, splitLinesKeepEnds } from "@youni/onnxweb-core/core";

import type { TranslatedChunk, TranslationResult } from "./translator.js";

/** Translates several texts from one source language. */
export type TranslateMany = (texts: string[], source: string | null) => Promise<TranslationResult[]>;
/** The language of a text (null or `und` when unknown). */
export type Detect = (text: string) => string | null | Promise<string | null>;

const LANG_PLACEHOLDERS = [/%\d+\$[a-zA-Z]/g, /%[sd]/g];

/** Extensions whose structure is kept; everything else is read as a document. */
export const STRUCTURED_EXTENSIONS: ReadonlySet<string> = new Set([".lang", ".srt", ".vtt", ".md"]);

export interface FileTranslation {
  text: string;
  original: string;
  results: TranslationResult[];
  metadata: Record<string, unknown>;
}

export function extensionOf(name: string): string {
  const base = name.split(/[\\/]/).pop() ?? "";
  const dot = base.lastIndexOf(".");
  return dot > 0 ? base.slice(dot).toLowerCase() : "";
}

export function stemOf(name: string): string {
  const base = name.split(/[\\/]/).pop() ?? "";
  const dot = base.lastIndexOf(".");
  return dot > 0 ? base.slice(0, dot) : base;
}

async function language(text: string, detect: Detect): Promise<string | null> {
  const found = await detect(text);
  return found === null || found === "und" ? null : found;
}

/** The text of an HTML page, one line per block (BeautifulSoup's `get_text("\n")`). */
export function htmlText(html: string): string {
  if (typeof DOMParser === "undefined") throw new Error("Reading .html needs a DOMParser (a browser or a worker with one)");
  const document = new DOMParser().parseFromString(html, "text/html");
  const parts: string[] = [];
  const walker = document.createTreeWalker(document.body ?? document.documentElement, NodeFilter.SHOW_TEXT);
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    const parent = node.parentElement?.tagName;
    if (parent === "SCRIPT" || parent === "STYLE") continue;
    parts.push(node.textContent ?? "");
  }
  return parts.join("\n");
}

/** Translate a file's text: `{text, original, results, metadata}`. */
export async function translateFileText(name: string, content: string, translate: TranslateMany, sourceLang: string | null, detect: Detect): Promise<FileTranslation> {
  const suffix = extensionOf(name);
  if (suffix === ".lang") return { original: content, ...(await translateLang(content, translate, sourceLang, detect)) };
  if (suffix === ".srt" || suffix === ".vtt") {
    const results: TranslationResult[] = [];
    const [text, cues] = await mapCues(content, async (texts) => {
      const source = sourceLang ?? (await language(texts.join(" "), detect));
      results.push(...(await translate(texts, source)));
      return results.map((r) => r.text);
    });
    return { text, original: content, results, metadata: { format: suffix.slice(1), cues } };
  }
  if (suffix === ".md") {
    const [body, metadata] = splitFrontMatter(content);
    const head = content.slice(0, content.length - body.length);
    const results = await translate([body], sourceLang ?? (await language(body, detect)));
    return { text: head + results[0]!.text, original: content, results, metadata: { format: "md", ...metadata } };
  }
  const html = suffix === ".html" || suffix === ".htm";
  const text = html ? htmlText(content) : content;
  const results = await translate([text], sourceLang ?? (await language(text, detect)));
  return { text: results[0]!.text, original: text, results, metadata: { format: html ? "html" : suffix.slice(1) || "txt" } };
}

/** The printf placeholders of a .lang value, sorted. */
export function langPlaceholders(text: string): string[] {
  return LANG_PLACEHOLDERS.flatMap((pattern) => text.match(pattern) ?? []).sort();
}

async function translateLang(content: string, translate: TranslateMany, sourceLang: string | null, detect: Detect): Promise<Omit<FileTranslation, "original">> {
  const lines = splitLinesKeepEnds(content);
  const values: [number, string, string][] = [];
  lines.forEach((line, index) => {
    const stripped = line.replace(/[\r\n]+$/, "");
    if (stripped && !stripped.startsWith("#") && stripped.includes("=")) {
      const at = stripped.indexOf("=");
      values.push([index, stripped.slice(0, at), stripped.slice(at + 1)]);
    }
  });
  if (values.length === 0) return { text: content, results: [], metadata: { format: "lang", lines: lines.length, kept: 0 } };
  const source = sourceLang ?? (await language(values.map(([, , v]) => v).join(" "), detect));
  const results = await translate(values.map(([, , v]) => v), source);
  let kept = 0;
  values.forEach(([index, key, value], i) => {
    const line = lines[index]!;
    const eol = line.slice(line.replace(/[\r\n]+$/, "").length);
    let translated = results[i]!.text;

    if (langPlaceholders(value).join("\0") !== langPlaceholders(translated).join("\0")) {
      translated = value;
      kept += 1;
    }
    lines[index] = `${key}=${translated}${eol}`;
  });
  return { text: lines.join(""), results, metadata: { format: "lang", lines: lines.length, kept } };
}

/** The result of a mixed text or a file: its text, and what its pieces went through. */
export function combineResults(text: string, original: string, results: readonly TranslationResult[], targetLang: string, started: number, metadata: Record<string, unknown>): TranslationResult {
  const languages: string[] = [];
  const models: string[] = [];
  const routes: string[][] = [];
  const backends: string[] = [];
  const chunks: TranslatedChunk[] = [];
  for (const result of results) {
    for (const lang of Array.isArray(result.sourceLang) ? result.sourceLang : [result.sourceLang]) if (!languages.includes(lang)) languages.push(lang);
    for (const model of Array.isArray(result.modelName) ? result.modelName : [result.modelName]) if (model && !models.includes(model)) models.push(model);
    if (result.route && !routes.some((r) => r.join("\0") === result.route!.join("\0"))) routes.push(result.route);
    if (result.backend && !backends.includes(result.backend)) backends.push(result.backend);
    for (const chunk of result.chunks) chunks.push({ ...chunk, id: chunks.length });
  }
  return {
    text,
    originalText: original,
    sourceLang: languages.length === 1 ? languages[0]! : languages.length > 0 ? languages : "und",
    targetLang,
    route: routes.length === 1 ? routes[0]! : null,
    backend: backends.length > 0 ? backends.join("+") : "none",
    modelName: models.length === 1 ? models[0]! : models,
    chunks,
    lostPlaceholders: results.reduce((s, r) => s + r.lostPlaceholders, 0),
    elapsedS: Math.round((performance.now() - started) / 1000 * 10000) / 10000,
    metadata,
  };
}

/** `[id, source, translation]` per piece; the whole text when there are none. */
function units(result: TranslationResult): [string, string, string][] {
  if (result.chunks.length > 0) return result.chunks.map((c) => [String(c.id), c.sourceText, c.translatedText]);
  return [["1", result.originalText, result.text]];
}

function poString(text: string): string {
  const escaped = text.replaceAll("\\", "\\\\").replaceAll('"', '\\"').replaceAll("\t", "\\t").replaceAll("\r", "\\r");
  const lines = escaped.split("\n");
  if (lines.length === 1) return `"${lines[0]}"`;

  const parts = lines.slice(0, -1).map((line) => `"${line}\\n"`);
  if (lines[lines.length - 1]) parts.push(`"${lines[lines.length - 1]}"`);
  return `""\n${parts.join("\n")}`;
}

/** GNU gettext: one msgid/msgstr pair per piece. */
export function toPo(result: TranslationResult): string {
  const header = `msgid ""\nmsgstr ""\n"Content-Type: text/plain; charset=UTF-8\\n"\n"Content-Transfer-Encoding: 8bit\\n"\n"Language: ${result.targetLang}\\n"\n`;
  const entries = [header];
  const seen = new Set<string>();
  for (const [id, source, target] of units(result)) {
    if (!source.trim() || seen.has(source)) continue;
    seen.add(source);
    entries.push(`#. piece ${id}\nmsgid ${poString(source)}\nmsgstr ${poString(target)}\n`);
  }
  return entries.join("\n");
}

const xmlEscape = (text: string, attribute = false) => {
  const escaped = text.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
  return attribute ? escaped.replaceAll('"', "&quot;").replaceAll("\n", "&#10;").replaceAll("\r", "&#13;").replaceAll("\t", "&#09;") : escaped;
};

/** XLIFF 1.2: one `<trans-unit>` per piece (ElementTree's output, indented by two). */
export function toXliff(result: TranslationResult, original = "translation"): string {
  const source = typeof result.sourceLang === "string" ? result.sourceLang : (result.sourceLang[0] ?? "und");
  const body = units(result)
    .map(([id, src, tgt]) => {
      const element = (tag: string, text: string) => (text ? `<${tag}>${xmlEscape(text)}</${tag}>` : `<${tag} />`);
      return `      <trans-unit id="${xmlEscape(id, true)}">\n        ${element("source", src)}\n        ${element("target", tgt)}\n      </trans-unit>`;
    })
    .join("\n");
  const attributes = `source-language="${xmlEscape(source, true)}" target-language="${xmlEscape(result.targetLang, true)}" datatype="plaintext" original="${xmlEscape(original, true)}"`;
  const inner = body ? `    <body>\n${body}\n    </body>` : "    <body />";
  return `<?xml version="1.0" encoding="utf-8"?>\n<xliff version="1.2" xmlns="urn:oasis:names:tc:xliff:document:1.2">\n  <file ${attributes}>\n${inner}\n  </file>\n</xliff>`;
}

/** The translation formats by name; the others are core's exporters. */
export const TRANSLATION_FORMATS: ReadonlySet<string> = new Set(["po", "pot", "xliff", "xlf"]);

/**
 * A translation rendered by `format`: the translated text as it is for `txt` and for the format of
 * the file it came from (`sourceSuffix`); `po`, `xliff`; else core's exporters (json, md with its
 * front matter, csv — one row per piece —, xml…).
 */
export function exportTranslation(result: TranslationResult, format: string, options: { sourceSuffix?: string; original?: string } = {}): string {
  const name = format.toLowerCase().replace(/^\./, "") || "txt";
  const own = (options.sourceSuffix ?? "").toLowerCase().replace(/^\./, "");
  if (name === "txt" || name === "text" || (own && name === own)) return result.text;
  if (name === "po" || name === "pot") return toPo(result);
  if (name === "xliff" || name === "xlf") return `${toXliff(result, options.original ?? "translation")}\n`;
  return exportString(result, name, name === "md" || name === "markdown" ? { frontMatter: true } : {});
}
