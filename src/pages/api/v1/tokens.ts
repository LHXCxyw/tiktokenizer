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

// 辅助函数：设置CORS头
function setCorsHeaders(res: NextApiResponse) {
    res.setHeader('Access-Control-Allow-Credentials', 'true');
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET,POST,OPTIONS');
    res.setHeader(
        'Access-Control-Allow-Headers',
        'X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version'
    );
}

export default async function handler(
    req: NextApiRequest,
    res: NextApiResponse
) {
    // 设置CORS头部
    setCorsHeaders(res);

    // 处理预检请求
    if (req.method === 'OPTIONS') {
        res.status(200).end();
        return;
    }

    // 检查HTTP方法
    if (req.method !== 'GET' && req.method !== 'POST') {
        return res.status(405).json({ error: "方法不允许，仅支持GET和POST请求" });
    }

    try {
        // 记录请求信息以便调试
        console.log("请求方法:", req.method);
        console.log("请求头:", JSON.stringify(req.headers, null, 2));
        console.log("Content-Type:", req.headers["content-type"]);
        console.log("请求体原始数据:", req.body);
        console.log("请求查询:", req.query);

        // 处理不同的Content-Type
        let requestData: any;
        if (req.method === 'POST') {
            const contentType = req.headers["content-type"];
            if (contentType && contentType.includes('application/json')) {
                // 请求体已经被Next.js解析为JSON
                requestData = req.body;
            } else if (contentType && contentType.includes('application/x-www-form-urlencoded')) {
                // 表单数据，已被Next.js解析
                requestData = req.body;
            } else if (typeof req.body === 'string') {
                // 尝试解析JSON字符串
                try {
                    requestData = JSON.parse(req.body);
                } catch (e) {
                    console.error("解析请求体失败:", e);
                    requestData = {};
                }
            } else {
                requestData = req.body || {};
            }
        } else {
            // GET请求使用查询参数
            requestData = req.query;
        }

        console.log("处理后的请求数据:", requestData);

        // 合并查询参数和请求体数据
        const data = { ...requestData, ...req.query };

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
