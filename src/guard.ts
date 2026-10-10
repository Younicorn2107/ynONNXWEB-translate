/**
 * What a translator must not translate: code, HTML, links, URLs, templates, Markdown emphasis. The
 * port of `ynonnx_translate.formatting.guard`.
 */

const TILDES = "~∼〜～≈";
const MASK_ANY = `[${TILDES}]\\s*\\d+\\s*[${TILDES}]`;
const MASK_RE = new RegExp(`[${TILDES}]\\s*(\\d+)\\s*[${TILDES}]`);

/**
 * Whether `text` holds placeholders and nothing to translate (the model would write a sentence of
 * its own).
 */
export function onlyMasks(text: string): boolean {
  const rest = text.replace(new RegExp(MASK_ANY, "g"), "");
  return rest !== text && ![...rest].some((ch) => /[\p{L}\p{N}]/u.test(ch));
}

/**
 * Spaces masking added, removed on restore: only before , . ) ] and after ( [ (French spaces before
 * ; : ! ?).
 */
const SPACE_BEFORE = new RegExp(`(${MASK_ANY})[ \\t]+([,.)\\]])`, "g");
const SPACE_AFTER = new RegExp(`([(\\[])[ \\t]+(${MASK_ANY})`, "g");

/** Protected spans, most specific first (a fenced block whole first; a link before a bare URL). */
export const PATTERNS: readonly [string, RegExp][] = [
  ["code", /```[\s\S]*?```/g],
  ["code", /~~~[\s\S]*?~~~/g],
  ["code", /`[^`\n]+`/g],
  ["html", /<!--[\s\S]*?-->/g],
  ["html", /<\/?[a-zA-Z][a-zA-Z0-9-]*(?:\s[^<>]*?)?\/?>/g],
  ["html", /&[a-zA-Z][a-zA-Z0-9]{1,10};|&#\d{1,6};/g],
  ["image", /!\[[^\]]*\]\([^)\s]*\)/g],
  ["link", /\[[^\]\n]*\]\([^)\s]*\)/g],

  ["bold", /\*\*[^*\n]*\s[^*\n]*\*\*/g],
  ["italic", /(?<!\*)\*[^*\n]*\s[^*\n]*\*(?!\*)/g],
  ["url", /(?:https?:\/\/|www\.)[^\s<>()[\]"'`]+(?<![.,;:!?])/g],
  ["url", /\b[\p{L}\p{N}_.+-]+@[\p{L}\p{N}_-]+\.[\p{L}\p{N}_.]{2,}\b/gu],
  ["template", /\{\{[^{}]*\}\}/g],
  ["template", /\{%[^%]*%\}/g],
  ["template", /\$\{[^}]*\}/g],
  ["template", /%\((\w+)\)[sdifr]/g],
  ["template", /%[sdifr]\b/g],
  ["template", /\{[a-zA-Z_][\p{L}\p{N}_.]*\}/gu],
];

/** Protected spans whose PART is prose: the label is translated separately and put back. */
export const INNER: Readonly<Record<string, RegExp>> = {
  link: /^\[([^\]]*)\]/,
  image: /^!\[([^\]]*)\]/,
  bold: /^\*\*([^*]+)\*\*$/,
  italic: /^\*([^*]+)\*$/,
};

export interface Masked {
  text: string;
  spans: Map<string, string>;
  kinds: Map<string, string>;
}

const indexOf = (token: string) => {
  const found = MASK_RE.exec(token);
  return found ? Number(found[1]) : -1;
};

export class FormattingGuard {
  /** Replace protected spans with `~N~`, spaced from what they touch (but not at a line's edge). */
  extract(text: string): Masked {
    const result: Masked = { text, spans: new Map(), kinds: new Map() };
    let counter = 0;
    for (const [kind, pattern] of PATTERNS) {
      const source = result.text;
      result.text = source.replace(pattern, (...args: unknown[]) => {
        const match = args[0] as string;
        const offset = args.find((a, i) => i > 0 && typeof a === "number") as number;
        const token = `~${counter}~`;
        result.spans.set(token, match);
        result.kinds.set(token, kind);
        counter += 1;
        const end = offset + match.length;
        const left = offset > 0 && !/\s/.test(source[offset - 1]!) ? " " : "";
        const right = end < source.length && !/\s/.test(source[end]!) ? " " : "";
        return `${left}${token}${right}`;
      });
    }
    result.text = result.text.replace(/[ \t]{2,}/g, " ").trim();
    return result;
  }

  /** The spans back; returns `[text, lost]`, the count of placeholders the model destroyed. */
  restore(text: string, masked: Masked): [string, number] {
    let restored = text.replace(SPACE_BEFORE, "$1$2").replace(SPACE_AFTER, "$1$2");
    let lost = 0;

    for (const token of [...masked.spans.keys()].sort((a, b) => indexOf(b) - indexOf(a))) {
      const original = masked.spans.get(token)!;
      if (restored.includes(token)) {
        restored = restored.split(token).join(original);
        continue;
      }
      const loose = new RegExp(`[${TILDES}]\\s*${indexOf(token)}\\s*[${TILDES}]`, "g");
      if (loose.test(restored)) {
        restored = restored.replace(loose, () => original);
        continue;
      }
      lost += 1;
    }
    return [restored.replace(/[ \t]{2,}/g, " ").trim(), lost];
  }

  /** `{token: label}` for the prose inside links, images and emphasis. */
  static innerTexts(masked: Masked): Map<string, string> {
    const labels = new Map<string, string>();
    for (const [token, content] of masked.spans) {
      const pattern = INNER[masked.kinds.get(token) ?? ""];
      const found = pattern?.exec(content);
      if (found?.[1]?.trim()) labels.set(token, found[1]);
    }
    return labels;
  }

  /** The spans rebuilt with their translated labels; an empty label keeps the original. */
  static withInner(masked: Masked, translated: ReadonlyMap<string, string>): Masked {
    for (const [token, label] of translated) {
      const pattern = INNER[masked.kinds.get(token) ?? ""];
      const previous = masked.spans.get(token);
      if (!pattern || previous === undefined || !label.trim()) continue;
      masked.spans.set(token, previous.replace(pattern, (m: string, group: string) => m.replace(group, () => label)));
    }
    return masked;
  }
}
