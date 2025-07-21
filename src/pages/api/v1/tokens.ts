import type { NextApiRequest, NextApiResponse } from "next";

import { z } from "zod";
import { type AllOptions, oaiEncodings, allModels } from "~/models";
import { createTokenizer } from "~/models/tokenizer";

// 配置API以支持解析JSON请求体
export const config = {
    api: {
        bodyParser: true,
    },
};

async function calculateTokens(model: AllOptions, text: string): Promise<ResponseType> {
    const tokenizer = await createTokenizer(model);
    const result = tokenizer.tokenize(text);

    return {
        name: result.name,
        tokens: result.tokens,
        count: result.count,
    };
}

const encoderSchema = z.object({
    text: z.string(),
    encoder: oaiEncodings,
    count_only: z.boolean().optional().default(false),
});

const modelSchema = z.object({
    text: z.string(),
    model: allModels,
    count_only: z.boolean().optional().default(false),
});

const requestSchema = z.union([encoderSchema, modelSchema]);

const fullResponseSchema = z.object({
    name: z.string(),
    tokens: z.number().array(),
    count: z.number(),
});

const countOnlyResponseSchema = z.object({
    name: z.string(),
    count: z.number(),
});

type FullResponseType = z.infer<typeof fullResponseSchema>;
type CountOnlyResponseType = z.infer<typeof countOnlyResponseSchema>;
type ResponseType = FullResponseType; // 内部使用，始终返回完整结果

export default async function handler(
    req: NextApiRequest,
    res: NextApiResponse
) {
    // 检查HTTP方法
    if (req.method !== 'GET' && req.method !== 'POST') {
        return res.status(405).json({ error: "方法不允许，仅支持GET和POST请求" });
    }

    try {
        // 记录请求信息以便调试
        console.log("请求方法:", req.method);
        console.log("请求头:", req.headers);
        console.log("请求体:", req.body);
        console.log("请求查询:", req.query);

        // 统一处理查询参数和请求体数据
        const data =
            typeof req.body === "object" ? { ...req.body, ...req.query } : req.query;

        console.log("合并后的数据:", data);

        // 解析布尔值参数
        if (typeof data.count_only === "string") {
            data.count_only = data.count_only.toLowerCase() === "true";
        }

        // 使用parse而非safeParse，与encode.ts保持一致
        try {
            const input = requestSchema.parse(data);

            let result: ResponseType | undefined;

            if ("encoder" in input) {
                result = await calculateTokens(input.encoder, input.text);
            } else {
                result = await calculateTokens(input.model, input.text);
            }

            if (!result) {
                return res.status(500).json({ error: "计算token失败" });
            }

            // 根据参数决定返回完整结果还是仅计数
            if (input.count_only) {
                const countOnlyResult: CountOnlyResponseType = {
                    name: result.name,
                    count: result.count,
                };
                return res.json(countOnlyResult);
            }

            // 返回完整结果
            return res.json(result);
        } catch (parseError) {
            console.error("请求解析错误:", parseError);
            return res.status(400).json({ error: "无效的请求参数", details: parseError });
        }
    } catch (error) {
        console.error("Token计算错误:", error);
        return res.status(500).json({ error: "服务器处理请求时出错" });
    }
} 
