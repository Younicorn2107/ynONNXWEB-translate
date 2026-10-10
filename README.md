# onnxweb-translate

Translation in the browser: OPUS-MT pairs and NLLB-200.

A pair model when one exists, NLLB-200 otherwise (or a pivot through English); the source language detected per segment; formatting, placeholders and Markdown kept through the translation; files (SRT, VTT, PO, XLIFF, JSON, CSV, Markdown) translated in place.

```ts
import * as ort from "onnxruntime-web";
import { TextTranslator } from "@youni/onnxweb-translate";

const translator = new TextTranslator({ ort, Tensor: ort.Tensor });
const result = await translator.translate("Bonjour tout le monde", { targetLang: "en" });
console.log(result.text);
```

Weights: OPUS-MT CC-BY-4.0 (attribution); NLLB-200 CC-BY-NC-4.0 (non-commercial).

## Install

Builds against `@youni/onnxweb-core`:

```sh
npm install
npm run build   # dist/
npm test        # unit tests (Node, onnxruntime-web on WebAssembly)
```

## Licence

Apache-2.0 for the code. Model weights keep their own licence (listed above where it matters).
