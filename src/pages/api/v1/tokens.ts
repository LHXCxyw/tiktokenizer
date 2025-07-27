import type { NextApiRequest, NextApiResponse } from "next";

import { z } from "zod";
import { type AllOptions, oaiEncodings, allModels } from "~/models";
import { createTokenizer } from "~/models/tokenizer";

// 配置API以支持解析JSON请求体，增加大小限制
export const config = {
    api: {
        bodyParser: {
            sizeLimit: '10mb', // 增加请求体大小限制
        },
        // 添加额外的响应超时时间
        responseLimit: false,
    },
};

async function calculateTokens(
    model: AllOptions,
    text: string,
    hostInfo?: string
): Promise<ResponseType> {
    const tokenizer = await createTokenizer(model, hostInfo ? { hostInfo } : undefined);
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

// 辅助函数：设置CORS头，支持多种来源
function setCorsHeaders(req: NextApiRequest, res: NextApiResponse) {
    // 获取请求的Origin
    const origin = req.headers.origin || '*';

    // 设置CORS头部
    res.setHeader('Access-Control-Allow-Credentials', 'true');
    res.setHeader('Access-Control-Allow-Origin', origin);
    res.setHeader('Access-Control-Allow-Methods', 'GET,POST,OPTIONS,PUT,DELETE,PATCH');
    res.setHeader(
        'Access-Control-Allow-Headers',
        'X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version, Authorization'
    );
}

export default async function handler(
    req: NextApiRequest,
    res: NextApiResponse
) {
    // 设置CORS头部
    setCorsHeaders(req, res);

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
        console.log("--------------------------------");
        console.log("收到新请求 - 时间:", new Date().toISOString());
        console.log("请求方法:", req.method);
        console.log("请求URL:", req.url);
        console.log("请求头:", JSON.stringify(req.headers, null, 2));
        console.log("Content-Type:", req.headers["content-type"]);

        // 安全地记录请求体，防止过大的请求体导致日志膨胀
        if (typeof req.body === 'object') {
            console.log("请求体类型:", typeof req.body, "是否为空:", Object.keys(req.body).length === 0);
            if (Object.keys(req.body).length < 10) {
                console.log("请求体内容:", JSON.stringify(req.body, null, 2));
            } else {
                console.log("请求体过大，不完全显示");
            }
        } else if (typeof req.body === 'string') {
            console.log("请求体类型:", typeof req.body, "长度:", req.body.length);
            if (req.body.length < 100) {
                console.log("请求体内容:", req.body);
            } else {
                console.log("请求体内容前100字符:", req.body.substring(0, 100) + "...");
            }
        } else {
            console.log("请求体类型:", typeof req.body);
        }

        console.log("请求查询:", req.query);

        // 处理不同的Content-Type
        let requestData: any = {};

        // 尝试从查询参数中获取数据
        if (Object.keys(req.query).length > 0) {
            Object.assign(requestData, req.query);
        }

        // 尝试从请求体中获取数据
        if (req.method === 'POST') {
            const contentType = String(req.headers["content-type"] || "").toLowerCase();

            if (contentType.includes('application/json')) {
                // 请求体已经被Next.js解析为JSON
                if (typeof req.body === 'object' && req.body !== null) {
                    Object.assign(requestData, req.body);
                } else if (typeof req.body === 'string') {
                    try {
                        const parsedBody = JSON.parse(req.body);
                        Object.assign(requestData, parsedBody);
                    } catch (e) {
                        console.error("解析JSON字符串请求体失败:", e);
                    }
                }
            } else if (contentType.includes('application/x-www-form-urlencoded')) {
                // 表单数据，已被Next.js解析
                if (typeof req.body === 'object' && req.body !== null) {
                    Object.assign(requestData, req.body);
                }
            } else if (contentType.includes('text/plain')) {
                // 纯文本请求体，尝试作为JSON解析
                if (typeof req.body === 'string') {
                    try {
                        const parsedBody = JSON.parse(req.body);
                        Object.assign(requestData, parsedBody);
                    } catch (e) {
                        console.error("解析纯文本请求体为JSON失败:", e);
                        // 如果没有指定model或encoder，尝试将整个文本作为text参数
                        if (!requestData.model && !requestData.encoder) {
                            requestData.text = req.body;
                            // 默认使用gpt-4o模型
                            requestData.model = "gpt-4o";
                        }
                    }
                }
            } else if (typeof req.body === 'object' && req.body !== null) {
                // 其他类型，但请求体已被解析为对象
                Object.assign(requestData, req.body);
            } else if (typeof req.body === 'string') {
                // 其他类型字符串，尝试解析为JSON
                try {
                    const parsedBody = JSON.parse(req.body);
                    Object.assign(requestData, parsedBody);
                } catch (e) {
                    console.error("解析未知类型的字符串请求体失败:", e);
                }
            }
        }

        console.log("合并后的数据:", requestData);

        // 解析布尔值参数
        if (typeof requestData.count_only === "string") {
            requestData.count_only = requestData.count_only.toLowerCase() === "true";
        }

        // 提取主机信息用于服务器端环境配置
        const hostInfo = req.headers.host ?
            `${req.headers['x-forwarded-proto'] || 'http'}://${req.headers.host}` :
            undefined;

        if (hostInfo) {
            console.log(`提取到主机信息: ${hostInfo}`);
        } else {
            console.log("无法提取主机信息，将使用默认配置");
        }

        // 使用parse而非safeParse，与encode.ts保持一致
        try {
            // 尝试解析请求数据
            const input = requestSchema.parse(requestData);

            let result: ResponseType | undefined;

            if ("encoder" in input) {
                console.log(`使用编码器 ${input.encoder} 计算token...`);
                result = await calculateTokens(input.encoder, input.text, hostInfo);
            } else {
                console.log(`使用模型 ${input.model} 计算token...`);
                result = await calculateTokens(input.model, input.text, hostInfo);
            }

            if (!result) {
                console.error("计算token失败，结果为空");
                return res.status(500).json({ error: "计算token失败" });
            }

            // 根据参数决定返回完整结果还是仅计数
            if (input.count_only) {
                const countOnlyResult: CountOnlyResponseType = {
                    name: result.name,
                    count: result.count,
                };
                console.log("返回仅计数结果:", countOnlyResult);
                return res.json(countOnlyResult);
            }

            // 返回完整结果
            console.log("返回完整结果, token数量:", result.count);
            return res.json(result);
        } catch (parseError: any) {
            console.error("请求解析错误:", parseError);
            return res.status(400).json({
                error: "无效的请求参数",
                message: parseError.message || "未知错误",
                details: parseError
            });
        }
    } catch (error: any) {
        console.error("Token计算错误:", error);
        return res.status(500).json({
            error: "服务器处理请求时出错",
            message: error.message || "未知错误"
        });
    }
} 
