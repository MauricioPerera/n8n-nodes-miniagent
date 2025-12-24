/**
 * Gemini API Provider
 * Uses Google's Generative AI REST API directly (no SDK)
 * Supports function calling and embeddings
 */

import type {
	LLMProvider,
	LLMResponse,
	LLMConfig,
	Message,
	ToolDefinition,
	ToolCall,
	EmbeddingConfig,
} from './LLMProvider';
import { generateToolCallId, toGeminiTools } from './LLMProvider';

const GEMINI_API_BASE = 'https://generativelanguage.googleapis.com/v1beta';

interface GeminiContent {
	role: 'user' | 'model';
	parts: GeminiPart[];
}

interface GeminiPart {
	text?: string;
	functionCall?: {
		name: string;
		args: Record<string, unknown>;
	};
	functionResponse?: {
		name: string;
		response: {
			result: string;
		};
	};
}

interface GeminiResponse {
	candidates?: Array<{
		content: {
			parts: GeminiPart[];
			role: string;
		};
		finishReason: string;
	}>;
	usageMetadata?: {
		promptTokenCount: number;
		candidatesTokenCount: number;
		totalTokenCount: number;
	};
	error?: {
		message: string;
		code: number;
	};
}

interface GeminiEmbeddingResponse {
	embedding?: {
		values: number[];
	};
	error?: {
		message: string;
		code: number;
	};
}

interface GeminiBatchEmbeddingResponse {
	embeddings?: Array<{
		values: number[];
	}>;
	error?: {
		message: string;
		code: number;
	};
}

export class GeminiProvider implements LLMProvider {
	private apiKey: string;

	constructor(apiKey: string) {
		this.apiKey = apiKey;
	}

	supportsEmbeddings(): boolean {
		return true;
	}

	async chat(
		messages: Message[],
		tools?: ToolDefinition[],
		config?: LLMConfig,
	): Promise<LLMResponse> {
		const model = config?.model || 'gemini-1.5-flash';
		const url = `${GEMINI_API_BASE}/models/${model}:generateContent?key=${this.apiKey}`;

		// Convert messages to Gemini format
		const contents = this.convertMessages(messages);

		// Build request body
		const body: Record<string, unknown> = {
			contents,
			generationConfig: {
				temperature: config?.temperature ?? 0.7,
				maxOutputTokens: config?.maxTokens ?? 4096,
				topP: config?.topP ?? 0.95,
			},
		};

		// Add tools if provided
		if (tools && tools.length > 0) {
			body.tools = toGeminiTools(tools);
		}

		// Make the request
		const response = await fetch(url, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
			},
			body: JSON.stringify(body),
		});

		if (!response.ok) {
			const errorText = await response.text();
			throw new Error(`Gemini API error: ${response.status} - ${errorText}`);
		}

		const data = (await response.json()) as GeminiResponse;

		if (data.error) {
			throw new Error(`Gemini API error: ${data.error.message}`);
		}

		return this.parseResponse(data);
	}

	async embed(
		texts: string[],
		config?: EmbeddingConfig,
	): Promise<number[][]> {
		const model = config?.model || 'text-embedding-004';

		// Gemini supports batch embedding
		if (texts.length === 1) {
			return [await this.embedSingle(texts[0], model)];
		}

		// For multiple texts, use batch endpoint
		const url = `${GEMINI_API_BASE}/models/${model}:batchEmbedContents?key=${this.apiKey}`;

		const requests = texts.map((text) => ({
			model: `models/${model}`,
			content: {
				parts: [{ text }],
			},
		}));

		const response = await fetch(url, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
			},
			body: JSON.stringify({ requests }),
		});

		if (!response.ok) {
			const errorText = await response.text();
			throw new Error(`Gemini Embeddings API error: ${response.status} - ${errorText}`);
		}

		const data = (await response.json()) as GeminiBatchEmbeddingResponse;

		if (data.error) {
			throw new Error(`Gemini Embeddings API error: ${data.error.message}`);
		}

		if (!data.embeddings) {
			throw new Error('Gemini Embeddings API returned no embeddings');
		}

		return data.embeddings.map((e) => e.values);
	}

	private async embedSingle(text: string, model: string): Promise<number[]> {
		const url = `${GEMINI_API_BASE}/models/${model}:embedContent?key=${this.apiKey}`;

		const response = await fetch(url, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
			},
			body: JSON.stringify({
				model: `models/${model}`,
				content: {
					parts: [{ text }],
				},
			}),
		});

		if (!response.ok) {
			const errorText = await response.text();
			throw new Error(`Gemini Embeddings API error: ${response.status} - ${errorText}`);
		}

		const data = (await response.json()) as GeminiEmbeddingResponse;

		if (data.error) {
			throw new Error(`Gemini Embeddings API error: ${data.error.message}`);
		}

		if (!data.embedding) {
			throw new Error('Gemini Embeddings API returned no embedding');
		}

		return data.embedding.values;
	}

	private convertMessages(messages: Message[]): GeminiContent[] {
		const contents: GeminiContent[] = [];
		let systemPrompt = '';

		for (const msg of messages) {
			if (msg.role === 'system') {
				// Gemini doesn't have system role, prepend to first user message
				systemPrompt = msg.content;
				continue;
			}

			if (msg.role === 'user') {
				const text = systemPrompt ? `${systemPrompt}\n\n${msg.content}` : msg.content;
				contents.push({
					role: 'user',
					parts: [{ text }],
				});
				systemPrompt = ''; // Clear after using
			} else if (msg.role === 'assistant') {
				const parts: GeminiPart[] = [];

				if (msg.content) {
					parts.push({ text: msg.content });
				}

				if (msg.toolCalls) {
					for (const tc of msg.toolCalls) {
						parts.push({
							functionCall: {
								name: tc.name,
								args: tc.arguments,
							},
						});
					}
				}

				contents.push({
					role: 'model',
					parts,
				});
			} else if (msg.role === 'tool') {
				// Find the tool call this is responding to
				const toolCallId = msg.toolCallId || 'unknown';
				const prevMsg = messages.find(
					(m) => m.toolCalls?.some((tc) => tc.id === toolCallId),
				);
				const toolName = prevMsg?.toolCalls?.find((tc) => tc.id === toolCallId)?.name || 'function';

				contents.push({
					role: 'user',
					parts: [{
						functionResponse: {
							name: toolName,
							response: {
								result: msg.content,
							},
						},
					}],
				});
			}
		}

		return contents;
	}

	private parseResponse(data: GeminiResponse): LLMResponse {
		const candidate = data.candidates?.[0];

		if (!candidate) {
			return {
				content: null,
				toolCalls: [],
				finishReason: 'error',
			};
		}

		const toolCalls: ToolCall[] = [];
		let content: string | null = null;

		for (const part of candidate.content.parts) {
			if (part.text) {
				content = (content || '') + part.text;
			}
			if (part.functionCall) {
				toolCalls.push({
					id: generateToolCallId(),
					name: part.functionCall.name,
					arguments: part.functionCall.args,
				});
			}
		}

		// Determine finish reason
		let finishReason: LLMResponse['finishReason'] = 'stop';
		if (toolCalls.length > 0) {
			finishReason = 'tool_calls';
		} else if (candidate.finishReason === 'MAX_TOKENS') {
			finishReason = 'length';
		}

		return {
			content,
			toolCalls,
			finishReason,
			usage: data.usageMetadata ? {
				promptTokens: data.usageMetadata.promptTokenCount,
				completionTokens: data.usageMetadata.candidatesTokenCount,
				totalTokens: data.usageMetadata.totalTokenCount,
			} : undefined,
		};
	}
}
