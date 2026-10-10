import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { Router, type TranslationBackend } from "../../src/backends.js";
import { combineResults, langPlaceholders, toPo, toXliff, translateFileText } from "../../src/files.js";
import { FormattingGuard, onlyMasks } from "../../src/guard.js";
import { normalize, toIso, toNllbCode } from "../../src/languages.js";
import { abbreviated, cut, join as joinPieces, reassemble, segmentsOf, sentences, split, type TranslationResult } from "../../src/translator.js";

const here = dirname(fileURLToPath(import.meta.url));
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const f = JSON.parse(readFileSync(join(here, "..", "fixtures", "pure.json"), "utf-8")) as any;

const fromMap = (map: ReadonlyMap<string, string>) => Object.fromEntries(map);

/** A Python result (snake_case) as this port names it. */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function camel(result: any): TranslationResult {
  return {
    text: result.text,
    originalText: result.original_text,
    sourceLang: result.source_lang,
    targetLang: result.target_lang,
    route: result.route,
    backend: result.backend,
    modelName: result.model_name,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    chunks: result.chunks.map((c: any) => ({ id: c.id, sourceText: c.source_text, translatedText: c.translated_text, sourceLang: c.source_lang, targetLang: c.target_lang, route: c.route, charStart: c.char_start, charEnd: c.char_end })),
    lostPlaceholders: result.lost_placeholders,
    elapsedS: result.elapsed_s ?? 0,
    metadata: result.metadata,
  };
}

describe("languages, against ynonnx_translate", () => {
  it("normalizes and converts codes", () => {
    for (const { input, normalize: n, nllb, iso } of f.languages) {
      expect([normalize(input), toNllbCode(input), toIso(input)], input).toEqual([n, nllb, iso]);
    }
  });
});

describe("formatting guard", () => {
  const guard = new FormattingGuard();
  it("masks, restores and relabels as Python does", () => {
    for (const c of f.guard) {
      const masked = guard.extract(c.text);
      expect(masked.text, c.text).toBe(c.masked);
      expect(fromMap(masked.spans)).toEqual(c.spans);
      expect(fromMap(masked.kinds)).toEqual(c.kinds);
      expect(fromMap(FormattingGuard.innerTexts(masked))).toEqual(c.inner);
      expect(guard.restore(masked.text, masked)).toEqual(c.restored);
      expect(guard.restore(c.damaged, masked)).toEqual(c.damagedRestored);
      expect(onlyMasks(masked.text)).toBe(c.onlyMasks);
      const again = guard.extract(c.text);
      FormattingGuard.withInner(again, new Map([...FormattingGuard.innerTexts(again)].map(([k, v]) => [k, v.toUpperCase()])));
      expect(fromMap(again.spans)).toEqual(c.relabelled);
    }
  });
});

describe("splitting and reassembly", () => {
  it("splits sentences, minding abbreviations", () => {
    for (const { text, sentences: expected } of f.sentences) expect(sentences(text), text).toEqual(expected);
    for (const [word, expected] of Object.entries(f.abbreviated)) expect(abbreviated(word), word).toBe(expected);
  });

  it("cuts long sentences on the best boundary", () => {
    const long = f.cut["900"][0] as string;
    for (const [n, expected] of Object.entries(f.cut)) expect(cut(long, Number(n))).toEqual(expected);
    expect(split(`${f.sentences[0].text} ${long}`, 120)).toEqual(f.split);
  });

  it("joins pieces", () => {
    expect([["Un.", " Deux. ", "", "Trois."], ["一。", "二。"], []].map(joinPieces)).toEqual(f.join);
  });

  it("separates prose from structure and puts it back", () => {
    for (const { text, segments, pieces, reassembled } of f.blocks) {
      const got = segmentsOf(text);
      expect(got, JSON.stringify(text)).toEqual(segments);
      const split40 = got.map(([isText, , content]) => (isText ? split(content, 40) : []));
      expect(split40).toEqual(pieces);
      expect(reassemble(got, split40, split40.flat().map((p) => p.toUpperCase()))).toBe(reassembled);
    }
  });
});

describe("routing", () => {
  const pairs = new Set(["fr>en", "en>fr", "ja>en", "en>de", "de>en", "ko>fr"]);
  const stub = (name: "marian" | "nllb", all: boolean): TranslationBackend => ({
    name,
    multilingual: name === "nllb",
    supports: async (s, t) => all || pairs.has(`${s}>${t}`),
    translateBatch: async (texts) => [...texts],
    isLoaded: false,
    close: async () => {},
  });

  it("chooses the same path", async () => {
    for (const c of f.routes) {
      const backends = new Map<string, TranslationBackend>();
      if (c.mounted !== "nllb") backends.set("marian", stub("marian", false));
      if (c.mounted !== "marian") backends.set("nllb", stub("nllb", true));
      const router = new Router(backends);
      if (c.error) {
        await expect(router.route(c.source, c.target)).rejects.toThrow();
        continue;
      }
      const r = await router.route(c.source, c.target);
      expect({ steps: r.steps, backend: r.backend, reason: r.reason, pivot: r.pivot }, `${c.mounted} ${c.source}>${c.target}`).toEqual({
        steps: c.steps,
        backend: c.backend,
        reason: c.reason,
        pivot: c.pivot,
      });
    }
  });
});

describe("files", () => {
  const stub = (text: string) => (text.includes("drop") ? text.replaceAll("%s", "").replaceAll("%1$s", "").toUpperCase() : text.toUpperCase());
  const many = async (texts: string[], source: string | null): Promise<TranslationResult[]> =>
    texts.map((t) => ({
      text: stub(t),
      originalText: t,
      sourceLang: source ?? "und",
      targetLang: "fr",
      route: [source ?? "und", "fr"],
      backend: "helsinki",
      modelName: `stub-${source}`,
      chunks: [],
      lostPlaceholders: 0,
      elapsedS: 0,
      metadata: {},
    }));

  it("keeps the structure of lang, srt, vtt, md and txt", async () => {
    for (const c of f.files) {
      const content = (c.content as string).replaceAll("\r\n", "\n");
      const done = await translateFileText(c.name, content, many, null, () => "en");
      expect(done.text, c.name).toBe(c.text);
      expect(done.original).toBe(c.original);
      expect(done.metadata).toEqual(c.metadata);
      const combined = combineResults(done.text, done.original, done.results, "fr", performance.now(), done.metadata);
      const expected = camel(c.combined);
      expect({ ...combined, elapsedS: 0 }).toEqual({ ...expected, elapsedS: 0 });
    }
    for (const [text, expected] of Object.entries(f.langPlaceholders)) expect(langPlaceholders(text)).toEqual(expected);
  });

  it("renders PO and XLIFF as ynlibs does", () => {
    for (const c of f.exported) {
      const result = camel(c.result);
      expect(toPo(result)).toBe(c.po);
      expect(toXliff(result, "doc")).toBe(c.xliff);
    }
  });
});

