/**
 * TextTranslator — translation in the browser: opus-mt pairs and NLLB. The port of
 * `ynonnx_translate.TextTranslator` over `ynlibs.translation`:
 */

import {
  ConfigurationError,
  IndexedDbStore,
  LanguageDetector,
  MIN_SEGMENT_CHARS,
  ResultCache,
  ResumableRun,
  defaultLogger,
  type KeyValueStore,
  type LanguageSegment,
  type Logger,
} from "@youni/onnxweb-core/core";

import { MarianBackend, NLLBBackend, Router, type BackendOptions, type Route, type TranslationBackend } from "./backends.js";
import { combineResults, exportTranslation, extensionOf, stemOf, translateFileText } from "./files.js";
import { FormattingGuard, onlyMasks, type Masked } from "./guard.js";
import { toIso, toNllbCode } from "./languages.js";

export interface TranslatedChunk {
  id: number;
  sourceText: string;
  translatedText: string;
  sourceLang: string;
  targetLang: string;
  route: string[] | null;
  charStart: number;
  charEnd: number;
}

export interface TranslationResult {
  text: string;
  originalText: string;
  /** Detected or given; a list for a mixed text. */
  sourceLang: string | string[];
  targetLang: string;
  /** `[source, target]`, `[source, pivot, target]`, or null when nothing was translated. */
  route: string[] | null;
  /** `helsinki` (opus-mt), `nllb`, or `none`. */
  backend: string;
  modelName: string | string[];
  chunks: TranslatedChunk[];
  /** Protected spans (a link, a tag, a variable) missing from the translation. */
  lostPlaceholders: number;
  elapsedS: number;
  metadata: Record<string, unknown>;
}

/** Markdown line openers kept out of the model (NLLB reads `##` as song lyrics). */
const PREFIX = /^(\s*(?:[#]{1,6}\s+|[-*+]\s+|\d+[.)]\s+|>\s*)+)/;
const SENTENCE_END = /\n\s*\n|(?<=[.!?…])\s+|(?<=[。！？])/g;
const ABBREVIATIONS = new Set(["mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "vs", "fig", "approx", "mme", "mlle", "hr", "nr", "bzw", "ca", "sra", "vol", "pp"]);
const BOUNDARIES = [/\n\s*\n/g, /(?<=[.!?…])\s+/g, /(?<=[;:])\s+/g, /(?<=,)\s+/g];
/** Decoding settings the pack measured; not arguments of the API. */
const LENGTH_PENALTY = 1;
const NO_REPEAT_NGRAM = 3;
const BACKEND_NAMES: Readonly<Record<string, string>> = { marian: "helsinki", nllb: "nllb", none: "none" };

const round4 = (x: number) => Math.round(x * 10000) / 10000;

/** Whether the period closing `sentence` belongs to an abbreviation ("Dr.", "e.g.", "J."). */
export function abbreviated(sentence: string): boolean {
  if (!sentence.endsWith(".")) return false;
  const words = sentence.slice(0, -1).split(/\s+/).filter(Boolean);
  if (words.length === 0) return false;
  const last = words[words.length - 1]!.replace(/^[(["'«]+/, "");
  return ([...last].length === 1 && /\p{L}/u.test(last)) || last.includes(".") || ABBREVIATIONS.has(last.toLowerCase());
}

/** The sentences of `text`; a period after an abbreviation ends none. */
export function sentences(text: string): string[] {
  const out: string[] = [];
  let start = 0;
  for (const match of text.matchAll(SENTENCE_END)) {
    const candidate = text.slice(start, match.index);
    const isSpace = match[0].trim() === "" && !match[0].includes("\n");
    if (isSpace && abbreviated(candidate)) continue;
    out.push(candidate);
    start = match.index + match[0].length;
  }
  out.push(text.slice(start));
  return out.map((s) => s.trim()).filter(Boolean);
}

/** A sentence too long for the encoder cut on the best boundary available (in characters). */
export function cut(text: string, maxChars: number): string[] {
  const stripped = text.trim();
  if ([...stripped].length <= maxChars) return stripped ? [stripped] : [];
  const chunks: string[] = [];
  let rest = stripped;
  while (rest.length > maxChars) {
    let at = 0;
    for (const pattern of BOUNDARIES) {
      const found = [...rest.slice(0, maxChars).matchAll(pattern)];
      if (found.length > 0) {
        const last = found[found.length - 1]!;
        at = last.index + last[0].length;
        break;
      }
    }
    if (at <= 0) {
      const space = rest.lastIndexOf(" ", maxChars - 1);
      at = space > 0 ? space + 1 : maxChars;
    }
    chunks.push(rest.slice(0, at).trim());
    rest = rest.slice(at).trim();
  }
  if (rest) chunks.push(rest);
  return chunks.filter(Boolean);
}

export function split(text: string, maxChars: number): string[] {
  return sentences(text).flatMap((s) => cut(s, maxChars));
}

/** Prose apart from structure, losing nothing: `[isText, prefix, content]`. */
export function segmentsOf(text: string): [boolean, string, string][] {
  const out: [boolean, string, string][] = [];
  text.split(/(\s*\n\s*)/).forEach((part, rank) => {
    if (!part) return;
    if (rank % 2 === 1 || onlyMasks(part)) {
      out.push([false, "", part]);
      return;
    }
    const opener = PREFIX.exec(part)?.[1] ?? "";
    out.push([true, opener, part.slice(opener.length)]);
  });
  return out;
}

/** Translated sentences put back together: a space, none after a CJK stop. */
export function join(pieces: readonly string[]): string {
  let joined = "";
  for (const piece of pieces.map((p) => p.trim())) {
    if (!piece) continue;
    if (joined && !/[。！？]$/.test(joined)) joined += " ";
    joined += piece;
  }
  return joined;
}

export interface TextTranslatorOptions extends Omit<BackendOptions, "computeType"> {
  /**
   * `auto` (opus-mt pair when one exists, NLLB otherwise), `helsinki` (pairs only, pivots), `nllb`.
   */
  model?: "auto" | "helsinki" | "nllb" | string;
  /** `auto`: int8 (the lightest published) on WebAssembly, float32 where WebGPU runs the model. */
  computeType?: string;
  sourceLang?: string | null;
  targetLang?: string;
  beamSize?: number;
  preserveFormatting?: boolean;
  /** Ceiling on the tokens written per piece. */
  maxTokens?: number;
  maxCharsPerPiece?: number;
  checkpoint?: KeyValueStore | boolean;
  cache?: KeyValueStore | boolean;
  /** The language detector; loaded on first need (fastText lid.176) when not given. */
  detector?: LanguageDetector;
}

export class TextTranslator {
  readonly modelName: string;
  sourceLang: string | null;
  targetLang: string;
  readonly beamSize: number;
  readonly preserveFormatting: boolean;
  readonly maxTokens: number;
  readonly maxCharsPerPiece: number;
  readonly #guard = new FormattingGuard();
  readonly #backends = new Map<string, TranslationBackend>();
  readonly #router: Router;
  readonly #logger: Logger;
  readonly #checkpoint: KeyValueStore | null;
  readonly #cache: ResultCache | null;
  #detector: LanguageDetector | null;
  readonly #nllbRepo: string;

  constructor(readonly options: TextTranslatorOptions) {
    const name = (options.model ?? "auto").trim();
    const lowered = name.toLowerCase();
    this.modelName = lowered === "opus-mt" || lowered === "marian" ? "helsinki" : name;
    this.sourceLang = options.sourceLang ? toIso(options.sourceLang) : null;
    this.targetLang = toIso(options.targetLang ?? "en");
    this.beamSize = options.beamSize ?? 4;
    this.preserveFormatting = options.preserveFormatting ?? true;
    this.maxTokens = options.maxTokens ?? 512;
    this.maxCharsPerPiece = options.maxCharsPerPiece ?? 900;
    this.#logger = options.logger ?? defaultLogger;
    this.#checkpoint = options.checkpoint === true ? new IndexedDbStore() : options.checkpoint || null;
    this.#cache = options.cache === true ? new ResultCache(new IndexedDbStore(), "ynonnx-translate") : options.cache ? new ResultCache(options.cache, "ynonnx-translate") : null;
    this.#detector = options.detector ?? null;

    const shared: BackendOptions = { ...options, computeType: options.computeType ?? "auto" };
    this.#nllbRepo = lowered === "nllb" || lowered === "auto" || lowered === "helsinki" || lowered.startsWith("nllb-200-distilled-600") ? "Xenova/nllb-200-distilled-600M" : name;
    if (this.modelName !== "nllb" && !lowered.startsWith("nllb") && !name.includes("/")) this.#backends.set("marian", new MarianBackend(shared));
    if (this.modelName !== "helsinki") this.#backends.set("nllb", new NLLBBackend(shared, this.#nllbRepo));
    this.#router = new Router(this.#backends);
  }

  get isLoaded(): boolean {
    return [...this.#backends.values()].some((b) => b.isLoaded);
  }

  async #languageDetector(): Promise<LanguageDetector> {
    this.#detector ??= await LanguageDetector.load(this.options.store ? { store: this.options.store } : {});
    return this.#detector;
  }

  /** The languages a translation goes through: `[source, target]`, a pivot between, or `[]`. */
  async route(source: string, target: string): Promise<string[]> {
    const chosen = await this.#router.route(source, target);
    return chosen.steps.length === 0 ? [] : [chosen.steps[0]![0], ...chosen.steps.map((s) => s[1])];
  }

  /** The language of `text` (`und` when unknown), or its runs of one language. */
  async detectLanguage(text: string, perSegment = false): Promise<string | LanguageSegment[]> {
    const detector = await this.#languageDetector();
    return perSegment ? detector.detectSegments(text) : (detector.detect(text, 0) ?? "und");
  }

  /** Translate one text, or several (resumed from a checkpoint and the cache when set). */
  async translate(texts: string, options?: { sourceLang?: string | null; targetLang?: string; beamSize?: number; signal?: AbortSignal }): Promise<TranslationResult>;
  async translate(texts: readonly string[], options?: { sourceLang?: string | null; targetLang?: string; beamSize?: number; signal?: AbortSignal }): Promise<TranslationResult[]>;
  async translate(texts: string | readonly string[], options: { sourceLang?: string | null; targetLang?: string; beamSize?: number; signal?: AbortSignal } = {}): Promise<TranslationResult | TranslationResult[]> {
    const single = typeof texts === "string";
    const items = single ? [texts] : [...texts];
    for (const item of items) if (typeof item !== "string") throw new ConfigurationError(`translate() takes a text or a list of texts, got ${typeof item}`);
    const source = options.sourceLang ? toIso(options.sourceLang) : this.sourceLang;
    const target = options.targetLang ? toIso(options.targetLang) : this.targetLang;
    if (!target) throw new ConfigurationError("No target language: pass targetLang");
    const beam = options.beamSize ?? this.beamSize;
    if (beam < 1) throw new ConfigurationError(`beam_size must be at least 1, got ${beam}`);
    const run = await ResumableRun.open({
      checkpoint: this.#checkpoint !== null,
      ...(this.#checkpoint ? { store: this.#checkpoint } : {}),
      inputs: items,
      settings: { model: this.modelName, source, target, beamSize: beam, preserveFormatting: this.preserveFormatting, computeType: this.options.computeType ?? "auto", maxTokens: this.maxTokens },
      cache: this.#cache,
    });
    const results: TranslationResult[] = [];
    for (const [index, text] of items.entries()) {
      if (await run.has(index)) {
        results.push(run.get(index) as TranslationResult);
        continue;
      }
      let result = await this.#translateOne(text, source, target, beam, options.signal);
      if (!result.originalText) result = { ...result, originalText: text };
      await run.record(index, result);
      results.push(result);
    }
    await run.complete();
    return single ? results[0]! : results;
  }

  #untouched(text: string, source: string, target: string, started: number, metadata: Record<string, unknown> = {}): TranslationResult {
    return { text: text ?? "", originalText: text ?? "", sourceLang: source, targetLang: target, route: null, backend: "none", modelName: this.modelName, chunks: [], lostPlaceholders: 0, elapsedS: round4((performance.now() - started) / 1000), metadata };
  }

  async #translateOne(text: string, source: string | null, target: string, beam: number, signal?: AbortSignal): Promise<TranslationResult> {
    const started = performance.now();
    if (!(text ?? "").trim()) return this.#untouched("", source ?? "und", target, started);

    if (source !== null && text.trim().length < MIN_SEGMENT_CHARS) return this.#translateText(text, source, target, beam, started, signal);
    const segments = (await this.#languageDetector()).detectSegments(text);
    const toTranslate = segments.filter((s) => !(s.lang === target || (source !== null && s.lang !== source)));
    const kept = segments.filter((s) => !toTranslate.includes(s));
    if (toTranslate.length === 0) return this.#untouched(text, source ?? segments[0]?.lang ?? "und", target, started, { passthrough: true });
    if (toTranslate.every((s) => s.lang === "und") && kept.length === 0 && source === null) {
      this.#logger.warn(`Could not identify the language of '${text.slice(0, 60)}'; it is returned untranslated. Pass sourceLang to translate it.`);
      return this.#untouched(text, "und", target, started, { undetected: true });
    }
    if (kept.length > 0 || new Set(toTranslate.map((s) => s.lang)).size > 1) return this.#translateMixed(text, segments, toTranslate, target, beam, signal);
    return this.#translateText(text, source ?? toTranslate[0]!.lang, target, beam, started, signal);
  }

  /** Each segment from its own language; the others verbatim, in place. */
  async #translateMixed(text: string, segments: LanguageSegment[], toTranslate: LanguageSegment[], target: string, beam: number, signal?: AbortSignal): Promise<TranslationResult> {
    const started = performance.now();
    const parts: string[] = [];
    const results: TranslationResult[] = [];
    let cursor = 0;
    for (const segment of [...segments].sort((a, b) => a.start - b.start)) {
      parts.push(text.slice(cursor, segment.start));
      const span = text.slice(segment.start, segment.end);
      cursor = segment.end;
      if (!toTranslate.includes(segment) || segment.lang === "und") {
        parts.push(span);
        continue;
      }
      const result = await this.#translateText(span, segment.lang, target, beam, performance.now(), signal);
      parts.push(result.text);
      results.push(result);
    }
    parts.push(text.slice(cursor));
    return combineResults(parts.join(""), text, results, target, started, { mixedLanguages: true, passthroughSegments: segments.filter((s) => !toTranslate.includes(s)).length });
  }

  /** One text in one language: protect, split, route, run, reassemble. */
  async #translateText(text: string, source: string, target: string, beam: number, started: number, signal?: AbortSignal): Promise<TranslationResult> {
    const stripped = text.trim();
    const head = text.slice(0, text.length - text.trimStart().length);
    const tail = text.slice(text.trimEnd().length);
    const chosen = await this.#router.route(source, target);
    if (chosen.steps.length === 0) return this.#untouched(text, toIso(source), toIso(target), started);
    const masked: Masked | null = this.preserveFormatting ? this.#guard.extract(stripped) : null;
    const blocks = segmentsOf(masked ? masked.text : stripped);
    const pieces = blocks.map(([isText, , content]) => (isText ? split(content, this.maxCharsPerPiece) : []));
    const labels = masked ? FormattingGuard.innerTexts(masked) : new Map<string, string>();
    const keys = [...labels.keys()];
    const batch = [...pieces.flat(), ...keys.map((k) => labels.get(k)!)];
    const backend = this.#backends.get(chosen.backend)!;
    let current = batch;
    for (const [stepSource, stepTarget] of chosen.steps) {
      current = await backend.translateBatch(current, stepSource, stepTarget, { beamSize: beam, maxTokens: this.maxTokens, lengthPenalty: LENGTH_PENALTY, noRepeatNgramSize: NO_REPEAT_NGRAM, ...(signal ? { signal } : {}) });
    }
    const at = batch.length - keys.length;
    const translated = current.slice(0, at);
    if (masked && keys.length > 0) FormattingGuard.withInner(masked, new Map(keys.map((k, i) => [k, current[at + i]!])));
    let joined = reassemble(blocks, pieces, translated);
    let lost = 0;
    if (masked) {
      [joined, lost] = this.#guard.restore(joined, masked);
      if (lost) this.#logger.warn(`${lost} protected span(s) missing from the translation — a link, a tag or a variable disappeared`);
    }
    const route = [chosen.steps[0]![0], ...chosen.steps.map((s) => s[1])];
    const sources = pieces.flat();
    return {
      text: head + joined + tail,
      originalText: text,
      sourceLang: toIso(source),
      targetLang: toIso(target),
      route,
      backend: BACKEND_NAMES[chosen.backend] ?? chosen.backend,
      modelName: this.#modelIds(chosen),
      chunks: sources.map((piece, id) => ({ id, sourceText: piece, translatedText: translated[id] ?? "", sourceLang: toIso(source), targetLang: toIso(target), route, charStart: 0, charEnd: 0 })),
      lostPlaceholders: lost,
      elapsedS: round4((performance.now() - started) / 1000),
      metadata: { reason: chosen.reason },
    };
  }

  #modelIds(chosen: Route): string | string[] {
    if (chosen.backend !== "marian") return this.#nllbRepo;
    const marian = this.#backends.get("marian") as MarianBackend;
    const ids = chosen.steps.map(([a, b]) => marian.repoFor(a, b));
    return ids.length === 1 ? ids[0]! : ids;
  }

  /**
   * Translate a file, its structure kept (see `files.ts`): a `File`, or a name and its text. Pieces
   * come from one source language, the one given or the one detected on the whole file.
   */
  async translateFile(file: Blob & { name: string } | { name: string; text: string }, options: { sourceLang?: string | null; targetLang?: string; beamSize?: number; signal?: AbortSignal } = {}): Promise<TranslationResult> {
    const started = performance.now();
    const content = "text" in file && typeof file.text === "string" ? file.text : await (file as Blob).text();
    const target = options.targetLang ? toIso(options.targetLang) : this.targetLang;
    const given = options.sourceLang ? toIso(options.sourceLang) : this.sourceLang;
    const many = async (texts: string[], source: string | null) =>
      texts.length === 0 ? [] : this.translate(texts, { sourceLang: source, targetLang: target, ...(options.beamSize !== undefined ? { beamSize: options.beamSize } : {}), ...(options.signal ? { signal: options.signal } : {}) });
    const detect = async (text: string) => (await this.#languageDetector()).detect(text, 0);
    const done = await translateFileText(file.name, content, many, given, detect);
    return combineResults(done.text, done.original, done.results, target, started, done.metadata);
  }

  /**
   * A translation as a file's content: by `format` (txt, the source's own, po, xliff, json, md,
   * csv…).
   */
  static export(result: TranslationResult, format: string, options: { sourceName?: string } = {}): string {
    return exportTranslation(result, format, options.sourceName ? { sourceSuffix: extensionOf(options.sourceName), original: stemOf(options.sourceName) } : {});
  }

  getModelInfo(): Record<string, unknown> {
    return {
      modelName: this.modelName,
      backends: [...this.#backends.keys()].map((b) => BACKEND_NAMES[b]),
      nllbModel: this.#backends.has("nllb") ? this.#nllbRepo : null,
      computeType: this.options.computeType ?? "auto",
      sourceLang: this.sourceLang,
      targetLang: this.targetLang,
      isLoaded: this.isLoaded,
      nllbCode: toNllbCode(this.targetLang),
    };
  }

  async close(): Promise<void> {
    for (const backend of this.#backends.values()) await backend.close();
  }
}

/** The blocks rejoined with their separators and prefixes. */
export function reassemble(blocks: readonly [boolean, string, string][], pieces: readonly string[][], translated: readonly string[]): string {
  const out: string[] = [];
  let cursor = 0;
  blocks.forEach(([isText, prefix, content], i) => {
    if (!isText) {
      out.push(content);
      return;
    }
    const done = translated.slice(cursor, cursor + pieces[i]!.length);
    cursor += pieces[i]!.length;
    out.push(prefix + join(done));
  });
  return out.join("");
}
