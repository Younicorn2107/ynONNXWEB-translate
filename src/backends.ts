/** The two families, and the router between them. */

import { detectCapabilities, huggingFaceResolver, loadModel, type LoadOptions, type ModelBundle } from "@youni/onnxweb-core";
import {
  GenerationEngine,
  SentencePieceProcessor,
  Tokenizer,
  YnONNXError,
  defaultLogger,
  registerErrors,
  type Logger,
  type TensorFactory,
} from "@youni/onnxweb-core/core";

import { PIVOTS, sameLanguage, toIso, toNllbCode } from "./languages.js";

export class YnONNXTranslateError extends YnONNXError {
  static override readonly code: string = "YnONNXTranslateError";
}

/** No path between two languages, or a code the model does not know. */
export class LanguageNotSupportedError extends YnONNXTranslateError {
  static override readonly code: string = "LanguageNotSupportedError";
}

registerErrors(YnONNXTranslateError, LanguageNotSupportedError);

export const REPO_TEMPLATE = (s: string, t: string) => `Xenova/opus-mt-${s}-${t}`;
export const FALLBACK_TEMPLATE = (s: string, t: string) => `onnx-community/opus-mt-${s}-${t}`;
export const ORIGINAL_TEMPLATE = (s: string, t: string) => `Helsinki-NLP/opus-mt-${s}-${t}`;
export const DEFAULT_NLLB_REPO = "Xenova/nllb-200-distilled-600M";

export interface DecodeSettings {
  beamSize: number;
  maxTokens: number;
  lengthPenalty: number;
  noRepeatNgramSize: number;
  signal?: AbortSignal;
}

export interface TranslationBackend {
  readonly name: "marian" | "nllb";
  readonly multilingual: boolean;
  supports(source: string, target: string): Promise<boolean>;
  translateBatch(texts: readonly string[], source: string, target: string, settings: DecodeSettings): Promise<string[]>;
  readonly isLoaded: boolean;
  close(): Promise<void>;
}

export interface BackendOptions extends Omit<LoadOptions, "model" | "computeType" | "resolver"> {
  Tensor: TensorFactory;
  computeType?: string;
  endpoint?: string;
  logger?: Logger;
}

const hub = (options: BackendOptions) => options.endpoint ?? "https://huggingface.co";

interface Loaded {
  bundle: ModelBundle;
  engine: GenerationEngine;
  tokenizer: Tokenizer;
}

/**
 * `auto`: the int8 graphs (dynamic quantization, the lightest published) on WebAssembly; float32
 * when WebGPU will run them — int8 has no WebGPU kernel (MatMulInteger), every matmul would go back
 * to the CPU.
 */
export async function autoComputeType(backends: unknown): Promise<string> {
  const wantsWasm = backends === "wasm" || (Array.isArray(backends) && backends[0] === "wasm");
  if (wantsWasm) return "int8";
  const caps = await detectCapabilities();
  return caps.webgpu ? "float32" : "int8";
}

async function loadSeq2seq(repo: string, options: BackendOptions, extra: readonly string[] = []): Promise<Loaded> {
  const { Tensor, computeType, endpoint, ...rest } = options;
  const resolved = !computeType || computeType === "auto" ? await autoComputeType(rest.backends) : computeType;
  const bundle = await loadModel({
    extraFiles: ["tokenizer_config.json", "config.json", "generation_config.json", ...extra],
    ...rest,
    resolver: huggingFaceResolver({ repo, subfolder: "onnx", manifest: false, ...(endpoint ? { endpoint } : {}) }),
    model: { name: repo, computeType: resolved },
  });
  const json = bundle.assetText(bundle.manifest.tokenizerFile ?? "tokenizer.json");
  if (json === null) throw new YnONNXTranslateError(`${repo} came without a tokenizer.json`);
  const tokenizer = Tokenizer.fromJson(json, bundle.assetText("tokenizer_config.json"));
  return { bundle, engine: new GenerationEngine(bundle, Tensor, options.logger ? { logger: options.logger } : {}), tokenizer };
}

/** The decoder's first token: the config's, else `<pad>` (Marian) or `</s>` (NLLB). */
function decoderStart(loaded: Loaded, fallback: number): number {
  for (const file of ["generation_config.json", "config.json"]) {
    const text = loaded.bundle.assetText(file);
    if (!text) continue;
    try {
      const value = (JSON.parse(text) as { decoder_start_token_id?: number }).decoder_start_token_id;
      if (typeof value === "number") return value;
    } catch {
      // Not fatal: carry on.
    }
  }
  return fallback;
}

export class MarianBackend implements TranslationBackend {
  readonly name = "marian" as const;
  readonly multilingual = false;
  /**
   * Pairs kept loaded at once: a pivot holds two, three avoids reloading when a document
   * alternates.
   */
  readonly maxLoaded: number;
  readonly #pairs = new Map<string, Loaded & { pieces: SentencePieceProcessor; repo: string }>();
  readonly #published = new Map<string, string | null>();

  constructor(private readonly options: BackendOptions, maxLoaded = 3) {
    this.maxLoaded = Math.max(Math.trunc(maxLoaded), 1);
  }

  get isLoaded(): boolean {
    return this.#pairs.size > 0;
  }

  /** The repository serving a pair, or null (asked once per pair). */
  async #repoFor(source: string, target: string): Promise<string | null> {
    const key = `${toIso(source)}-${toIso(target)}`;
    if (this.#published.has(key)) return this.#published.get(key)!;
    let found: string | null = null;
    for (const repo of [REPO_TEMPLATE(toIso(source), toIso(target)), FALLBACK_TEMPLATE(toIso(source), toIso(target))]) {
      try {
        const response = await fetch(`${hub(this.options)}/api/models/${repo}`, { method: "GET" });
        if (response.ok) {
          found = repo;
          break;
        }
      } catch {
        // Not fatal: carry on.
      }
    }
    this.#published.set(key, found);
    return found;
  }

  async supports(source: string, target: string): Promise<boolean> {
    if (this.#pairs.has(`${toIso(source)}-${toIso(target)}`)) return true;
    return (await this.#repoFor(source, target)) !== null;
  }

  async #load(source: string, target: string) {
    const key = `${toIso(source)}-${toIso(target)}`;
    const held = this.#pairs.get(key);
    if (held) {
      this.#pairs.delete(key);
      this.#pairs.set(key, held);
      return held;
    }
    const repo = await this.#repoFor(source, target);
    if (!repo) throw new LanguageNotSupportedError(`no published opus-mt model for ${toIso(source)} -> ${toIso(target)}. Use a pivot through English, or NLLB, which covers every pair.`);

    const loaded = await loadSeq2seq(repo, this.options, ["source.spm"]);
    const spmUrl = `${hub(this.options)}/${ORIGINAL_TEMPLATE(toIso(source), toIso(target))}/resolve/main/source.spm`;
    let spm = loaded.bundle.asset("source.spm");
    if (!spm) {
      const response = await fetch(spmUrl);
      if (!response.ok) throw new YnONNXTranslateError(`cannot fetch source.spm (${spmUrl}): without it the input is cut unlike training and the translation degrades`);
      spm = await response.arrayBuffer();
    }
    const entry = { ...loaded, pieces: new SentencePieceProcessor(new Uint8Array(spm)), repo };
    this.#pairs.set(key, entry);
    while (this.#pairs.size > this.maxLoaded) {
      const [oldest, value] = this.#pairs.entries().next().value!;
      this.#pairs.delete(oldest);
      await value.bundle.close();
    }
    return entry;
  }

  /**
   * The pieces of the pair's SentencePiece model, mapped onto the shared vocabulary, then `</s>`.
   */
  #encode(entry: { pieces: SentencePieceProcessor; tokenizer: Tokenizer }, text: string): number[] {
    const unknown = entry.tokenizer.tokenToId("<unk>") ?? 0;
    const ids = entry.pieces.encodePieces(text).map((piece) => entry.tokenizer.tokenToId(piece) ?? unknown);
    return [...ids, entry.tokenizer.tokenToId("</s>") ?? 0];
  }

  async translateBatch(texts: readonly string[], source: string, target: string, settings: DecodeSettings): Promise<string[]> {
    const entry = await this.#load(source, target);
    const pad = entry.tokenizer.tokenToId("<pad>");
    const eos = entry.tokenizer.tokenToId("</s>");
    const start = decoderStart(entry, pad ?? 0);
    const out: string[] = [];
    for (const text of texts) {
      const stripped = (text ?? "").trim();
      if (!stripped) {
        out.push("");
        continue;
      }
      const ids = this.#encode(entry, stripped);
      const result = await this.#run(entry, ids, [start], settings, eos, pad, pad !== null ? [pad] : []);
      out.push(entry.tokenizer.decode(result, { skipSpecialTokens: true }).trim());
    }
    return out;
  }

  async #run(entry: Loaded, ids: number[], prompt: number[], settings: DecodeSettings, eos: number | null, pad: number | null, suppress: number[]): Promise<number[]> {
    return runOne(entry, this.options.Tensor, ids, prompt, settings, eos, pad, suppress);
  }

  async close(): Promise<void> {
    for (const entry of this.#pairs.values()) await entry.bundle.close();
    this.#pairs.clear();
  }

  /** The repositories serving the pairs of a route, as the result reports them. */
  repoFor(source: string, target: string): string {
    return REPO_TEMPLATE(toIso(source), toIso(target));
  }
}

async function runOne(entry: Loaded, Tensor: TensorFactory, ids: number[], prompt: number[], settings: DecodeSettings, eos: number | null, pad: number | null, suppress: number[]): Promise<number[]> {
  const shape = [1, ids.length];
  const encoded = await entry.engine.encode({
    input_ids: new Tensor("int64", BigInt64Array.from(ids, (i) => BigInt(i)), shape),
    attention_mask: new Tensor("int64", new BigInt64Array(ids.length).fill(1n), shape),
  });
  const result = await entry.engine.generate({
    prompts: [prompt],
    encoderOutput: encoded,
    strategy: settings.beamSize > 1 ? "beam" : "greedy",
    beamSize: settings.beamSize,
    lengthPenalty: settings.lengthPenalty,
    noRepeatNgramSize: settings.noRepeatNgramSize,
    maxNewTokens: settings.maxTokens,
    temperature: 0,
    eosTokenId: eos,
    padTokenId: pad,
    suppressTokens: suppress,
    ...(settings.signal ? { signal: settings.signal } : {}),
  });
  return [...(result.sequences[0] ?? [])];
}

export class NLLBBackend implements TranslationBackend {
  readonly name = "nllb" as const;
  readonly multilingual = true;
  #loaded: Loaded | null = null;

  constructor(private readonly options: BackendOptions, readonly repo = DEFAULT_NLLB_REPO) {}

  get isLoaded(): boolean {
    return this.#loaded !== null;
  }

  /**
   * On WebAssembly only. Measured on 2026-10-05 with onnxruntime-web 1.29: under WebGPU the M2M100
   * graphs compute garbage at every precision (int8, float16, q4f16) — the encoder's output alone
   * is off by 3.5 —, with the encoder on WASM too. Marian and Whisper are not affected. Served on
   * WASM, where the output is Python's, until a runtime release fixes it.
   */
  async #load(): Promise<Loaded> {
    if (!this.#loaded) {
      const asked = this.options.backends;
      if (asked !== undefined && asked !== "wasm" && asked !== "auto") {
        (this.options.logger ?? defaultLogger).warn(`NLLB runs on WebAssembly: onnxruntime-web computes M2M100 wrongly under WebGPU (backends ${JSON.stringify(asked)} ignored)`);
      }
      this.#loaded = await loadSeq2seq(this.repo, { ...this.options, backends: "wasm" });
    }
    return this.#loaded;
  }

  /** Any pair whose two codes are NLLB languages, answered without loading. */
  async supports(source: string, target: string): Promise<boolean> {
    if (this.#loaded) return this.#languageId(source) !== null && this.#languageId(target) !== null;
    return Boolean(toNllbCode(source)) && Boolean(toNllbCode(target));
  }

  #languageId(language: string): number | null {
    const code = toNllbCode(language);
    return code && this.#loaded ? this.#loaded.tokenizer.tokenToId(code) : null;
  }

  async translateBatch(texts: readonly string[], source: string, target: string, settings: DecodeSettings): Promise<string[]> {
    const loaded = await this.#load();
    const sourceId = this.#languageId(source);
    const targetId = this.#languageId(target);
    if (sourceId === null || targetId === null) {
      throw new LanguageNotSupportedError(`'${sourceId === null ? source : target}' is not an NLLB-200 language; known two-letter ISO codes are converted automatically.`);
    }
    const eos = loaded.tokenizer.tokenToId("</s>") ?? 2;
    const pad = loaded.tokenizer.tokenToId("<pad>") ?? 1;
    const start = decoderStart(loaded, eos);
    const out: string[] = [];

    for (const text of texts) {
      const stripped = (text ?? "").trim();
      if (!stripped) {
        out.push("");
        continue;
      }
      let ids = loaded.tokenizer.encode(stripped, { addSpecialTokens: true });

      const first = ids[0] !== undefined ? loaded.tokenizer.idToToken(ids[0]) : null;
      if (first && first.length === 8 && first[3] === "_") ids = [sourceId, ...ids.slice(1)];
      else ids = [sourceId, ...ids];
      const result = await runOne(loaded, this.options.Tensor, ids, [start, targetId], settings, eos, pad, []);
      out.push(loaded.tokenizer.decode(result, { skipSpecialTokens: true }).trim());
    }
    return out;
  }

  async close(): Promise<void> {
    await this.#loaded?.bundle.close();
    this.#loaded = null;
  }
}

export interface Route {
  steps: [string, string][];
  backend: "marian" | "nllb" | "none";
  reason: string;
  pivot: string | null;
}

/** The path from one language to another, asking the mounted backends. */
export class Router {
  constructor(
    readonly backends: ReadonlyMap<string, TranslationBackend>,
    readonly pivots: readonly string[] = PIVOTS,
  ) {}

  async #first(source: string, target: string, multilingual: boolean): Promise<TranslationBackend | null> {
    for (const backend of this.backends.values()) {
      if (backend.multilingual !== multilingual) continue;
      try {
        if (await backend.supports(source, target)) return backend;
      } catch {
        // Not fatal: carry on.
      }
    }
    return null;
  }

  async route(source: string, target: string): Promise<Route> {
    const src = toIso(source);
    const tgt = toIso(target);
    if (!src || !tgt) throw new LanguageNotSupportedError(`a language is missing: source='${source}', target='${target}'`);
    if (sameLanguage(src, tgt)) return { steps: [], backend: "none", reason: "same input and output language", pivot: null };
    const direct = await this.#first(src, tgt, false);
    if (direct) return { steps: [[src, tgt]], backend: direct.name, reason: "direct pair", pivot: null };
    const multilingual = await this.#first(src, tgt, true);
    if (multilingual) return { steps: [[src, tgt]], backend: multilingual.name, reason: "no direct pair; multilingual model", pivot: null };
    for (const pivot of this.pivots) {
      if (sameLanguage(pivot, src) || sameLanguage(pivot, tgt)) continue;
      const outbound = await this.#first(src, pivot, false);
      const inbound = await this.#first(pivot, tgt, false);
      if (outbound && inbound && outbound === inbound) {
        return { steps: [[src, pivot], [pivot, tgt]], backend: outbound.name, reason: `no direct pair; pivot through '${pivot}'`, pivot };
      }
    }
    throw new LanguageNotSupportedError(`no path from ${src} to ${tgt}: no direct pair, no pivot through ${this.pivots.join(", ")}, and no multilingual model is mounted. Mount NLLB, which covers every pair.`);
  }
}
