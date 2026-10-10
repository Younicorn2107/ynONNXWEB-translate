/**
 * Language codes and conversion between their conventions: ISO 639-1 (`fr`), NLLB (`fra_Latn`,
 * language AND script), opus-mt pair names. The port of `ynonnx_translate.constants.languages`.
 */

export const ISO_TO_NLLB: Readonly<Record<string, string>> = {
  af: "afr_Latn", am: "amh_Ethi", ar: "arb_Arab", az: "azj_Latn",
  be: "bel_Cyrl", bg: "bul_Cyrl", bn: "ben_Beng", bs: "bos_Latn",
  ca: "cat_Latn", cs: "ces_Latn", cy: "cym_Latn", da: "dan_Latn",
  de: "deu_Latn", el: "ell_Grek", en: "eng_Latn", eo: "epo_Latn",
  es: "spa_Latn", et: "est_Latn", eu: "eus_Latn", fa: "pes_Arab",
  fi: "fin_Latn", fr: "fra_Latn", ga: "gle_Latn", gl: "glg_Latn",
  gu: "guj_Gujr", ha: "hau_Latn", he: "heb_Hebr", hi: "hin_Deva",
  hr: "hrv_Latn", hu: "hun_Latn", hy: "hye_Armn", id: "ind_Latn",
  is: "isl_Latn", it: "ita_Latn", ja: "jpn_Jpan", jv: "jav_Latn",
  ka: "kat_Geor", kk: "kaz_Cyrl", km: "khm_Khmr", kn: "kan_Knda",
  ko: "kor_Hang", lo: "lao_Laoo", lt: "lit_Latn", lv: "lvs_Latn",
  mk: "mkd_Cyrl", ml: "mal_Mlym", mn: "khk_Cyrl", mr: "mar_Deva",
  ms: "zsm_Latn", mt: "mlt_Latn", my: "mya_Mymr", ne: "npi_Deva",
  nl: "nld_Latn", no: "nob_Latn", pa: "pan_Guru", pl: "pol_Latn",
  ps: "pbt_Arab", pt: "por_Latn", ro: "ron_Latn", ru: "rus_Cyrl",
  si: "sin_Sinh", sk: "slk_Latn", sl: "slv_Latn", so: "som_Latn",
  sq: "als_Latn", sr: "srp_Cyrl", sv: "swe_Latn", sw: "swh_Latn",
  ta: "tam_Taml", te: "tel_Telu", th: "tha_Thai", tl: "tgl_Latn",
  tr: "tur_Latn", uk: "ukr_Cyrl", ur: "urd_Arab", uz: "uzn_Latn",
  vi: "vie_Latn", yo: "yor_Latn", zh: "zho_Hans", zu: "zul_Latn",
};

/** Language names accepted in place of a code, in English and French. */
export const NAME_TO_ISO: Readonly<Record<string, string>> = {
  français: "fr", francais: "fr", french: "fr",
  anglais: "en", english: "en",
  allemand: "de", german: "de", deutsch: "de",
  espagnol: "es", spanish: "es", español: "es",
  italien: "it", italian: "it", italiano: "it",
  portugais: "pt", portuguese: "pt",
  néerlandais: "nl", neerlandais: "nl", dutch: "nl",
  russe: "ru", russian: "ru",
  chinois: "zh", chinese: "zh",
  japonais: "ja", japanese: "ja",
  coréen: "ko", coreen: "ko", korean: "ko",
  arabe: "ar", arabic: "ar",
  polonais: "pl", polish: "pl",
  turc: "tr", turkish: "tr",
};

const TRADITIONAL = new Set(["hant", "tw", "hk", "mo"]);

/** Pivot languages, in order: English (opus-mt pairs almost always exist), then French. */
export const PIVOTS: readonly string[] = ["en", "fr"];

const isNllb = (code: string) => code.length === 8 && code[3] === "_";

/** A lowercase ISO 639-1 code from almost any spelling; a full NLLB code unchanged. */
export function normalize(language: string): string {
  const raw = (language ?? "").trim();
  if (!raw) return "";
  if (isNllb(raw)) return raw;
  const lower = raw.toLowerCase();
  if (NAME_TO_ISO[lower]) return NAME_TO_ISO[lower]!;
  const parts = lower.replace(/_/g, "-").split("-");
  if ((parts[0] === "zh" || parts[0] === "zho") && parts.slice(1).some((p) => TRADITIONAL.has(p))) return "zho_Hant";
  return parts[0]!.slice(0, 3);
}

export function toNllbCode(language: string): string {
  const n = normalize(language);
  if (isNllb(n)) return n;
  return ISO_TO_NLLB[n.slice(0, 2)] ?? "";
}

export function toIso(language: string): string {
  const n = normalize(language);
  if (isNllb(n)) {
    for (const [iso, nllb] of Object.entries(ISO_TO_NLLB)) if (nllb === n) return iso;
    return n.slice(0, 2);
  }
  return n.slice(0, 2);
}

export function sameLanguage(a: string, b: string): boolean {
  return Boolean(a) && Boolean(b) && toIso(a) === toIso(b);
}
