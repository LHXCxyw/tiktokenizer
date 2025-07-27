// 在导入 @xenova/transformers 之前设置环境变量
if (typeof window === "undefined") {
  // 服务器环境：禁用缓存相关功能以避免文件系统写入错误
  process.env.TRANSFORMERS_OFFLINE = "1";
  process.env.HF_HUB_DISABLE_TELEMETRY = "1";
}

import { hackModelsRemoveFirstToken } from "./index";
import { get_encoding, encoding_for_model, type Tiktoken } from "tiktoken";
import { oaiEncodings, oaiModels, openSourceModels } from ".";
import { PreTrainedTokenizer, env } from "@xenova/transformers";
import type { z } from "zod";
import {
  getHuggingfaceSegments,
  getTiktokenSegments,
  type Segment,
} from "~/utils/segments";

// Lambda 内存缓存：缓存已加载的 tokenizer 对象
const tokenizerCache = new Map<string, Tokenizer>();

export interface TokenizerResult {
  name: string;
  // Array<{ text: string; tokens: { id: number; idx: number }[] }> ?
  tokens: number[];
  segments?: Segment[];
  count: number;
}

export interface Tokenizer {
  name: string;
  tokenize(text: string): TokenizerResult;
  free?(): void;
}

export class TiktokenTokenizer implements Tokenizer {
  private enc: Tiktoken;
  name: string;
  constructor(model: z.infer<typeof oaiModels> | z.infer<typeof oaiEncodings>) {
    const isModel = oaiModels.safeParse(model);
    const isEncoding = oaiEncodings.safeParse(model);
    console.log(isModel.success, isEncoding.success, model)
    if (isModel.success) {

      if (
        model === "text-embedding-3-small" ||
        model === "text-embedding-3-large"
      ) {
        throw new Error("Model may be too new");
      }

      const enc =
        model === "gpt-3.5-turbo" || model === "gpt-4" || model === "gpt-4-32k"
          ? get_encoding("cl100k_base", {
            "<|im_start|>": 100264,
            "<|im_end|>": 100265,
            "<|im_sep|>": 100266,
          })
          : model === "gpt-4o"
            ? get_encoding("o200k_base", {
              "<|im_start|>": 200264,
              "<|im_end|>": 200265,
              "<|im_sep|>": 200266,
            })
            : // @ts-expect-error r50k broken?
            encoding_for_model(model);
      this.name = enc.name ?? model;
      this.enc = enc;
    } else if (isEncoding.success) {
      this.enc = get_encoding(isEncoding.data);
      this.name = isEncoding.data;
    } else {
      throw new Error("Invalid model or encoding");
    }
  }

  tokenize(text: string): TokenizerResult {
    const tokens = [...(this.enc?.encode(text, "all") ?? [])];
    return {
      name: this.name,
      tokens,
      segments: getTiktokenSegments(this.enc, text),
      count: tokens.length,
    };
  }

  free(): void {
    this.enc.free();
  }
}

export class OpenSourceTokenizer implements Tokenizer {
  constructor(private tokenizer: PreTrainedTokenizer, name?: string) {
    this.name = name ?? tokenizer.name;
  }

  name: string;

  static async load(
    model: z.infer<typeof openSourceModels>,
    hostInfo?: string
  ): Promise<PreTrainedTokenizer> {
    // 使用外部传入的主机信息，如果提供的话
    if (hostInfo) {
      console.log(`使用外部提供的主机信息: ${hostInfo}`);
      env.remoteHost = hostInfo;
    } else if (typeof window !== "undefined") {
      // 浏览器环境：使用当前页面的origin
      console.log("浏览器环境，使用当前页面origin");
      env.remoteHost = window.location.origin;
    } else {
      // 服务器环境但未提供主机信息：使用默认行为
      console.log("服务器环境且未提供主机信息，将使用默认远程主机");
    }
    env.remotePathTemplate = "/hf/{model}";
    // 在服务器环境中配置缓存策略
    if (typeof window === "undefined") {
      console.log("服务器环境：配置无缓存模式");
      // 完全禁用缓存以避免文件系统写入错误
      env.useBrowserCache = false;
      env.allowLocalModels = false;
    }
    const t = await PreTrainedTokenizer.from_pretrained(model, {
      // 最小化日志输出：只在关键状态时记录
      progress_callback: (progress: any) => {
        if (progress.status === 'initiate') {
          console.log(`开始加载 "${model}" ${progress.file}`);
        } else if (progress.status === 'done') {
          console.log(`完成加载 "${model}" ${progress.file}`);
        }
        // 忽略所有 'download' 和 'progress' 状态以减少日志
      },
      // 禁用本地文件优先机制，强制从远程下载
      local_files_only: false,
    });
    console.log("loaded tokenizer", model, t.name);
    return t;
  }

  tokenize(text: string): TokenizerResult {
    // const tokens = this.tokenizer(text);
    const tokens = this.tokenizer.encode(text);
    const removeFirstToken = (
      hackModelsRemoveFirstToken.options as string[]
    ).includes(this.name);
    return {
      name: this.name,
      tokens,
      segments: getHuggingfaceSegments(this.tokenizer, text, removeFirstToken),
      count: tokens.length,
    };
  }
}

export async function createTokenizer(
  name: string,
  options?: { hostInfo?: string }
): Promise<Tokenizer> {
  console.log("createTokenizer", name, options?.hostInfo ? `with hostInfo: ${options.hostInfo}` : "without hostInfo");

  // 🚀 检查 Lambda 内存缓存：如果已存在，直接返回（毫秒级速度）
  const cacheKey = `${name}_${options?.hostInfo || 'default'}`;
  if (tokenizerCache.has(cacheKey)) {
    console.log(`🚀 从缓存返回 tokenizer: ${name} (超快速度)`);
    return tokenizerCache.get(cacheKey)!;
  }

  const oaiEncoding = oaiEncodings.safeParse(name);
  if (oaiEncoding.success) {
    console.log("oaiEncoding", oaiEncoding.data);
    const tokenizer = new TiktokenTokenizer(oaiEncoding.data);
    tokenizerCache.set(cacheKey, tokenizer);
    return tokenizer;
  }
  const oaiModel = oaiModels.safeParse(name);
  if (oaiModel.success) {
    console.log("oaiModel", oaiModel.data);
    const tokenizer = new TiktokenTokenizer(oaiModel.data);
    tokenizerCache.set(cacheKey, tokenizer);
    return tokenizer;
  }

  const ossModel = openSourceModels.safeParse(name);
  if (ossModel.success) {
    console.log("loading tokenizer", ossModel.data);
    const huggingfaceTokenizer = await OpenSourceTokenizer.load(ossModel.data, options?.hostInfo);
    console.log("loaded tokenizer", name);
    const tokenizer = new OpenSourceTokenizer(huggingfaceTokenizer, name);

    // 💾 缓存到 Lambda 内存，后续调用将达到前端一样的超快速度
    tokenizerCache.set(cacheKey, tokenizer);
    console.log(`💾 已缓存 tokenizer: ${name} - 后续调用将超快`);

    return tokenizer;
  }
  throw new Error("Invalid model or encoding");
}
