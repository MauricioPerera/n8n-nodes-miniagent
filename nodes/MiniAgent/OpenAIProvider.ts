/**
 * OpenAI Compatible API Provider
 * Works with: OpenAI, OpenRouter, Groq, Together, Ollama, LM Studio, etc.
 * Uses the standard OpenAI chat completions format
 */

import type {
	LLMProvider,
	LLMResponse,
	LLMConfig,
	Message,
	ToolDefinition,
	ToolCall,
} from './LLMProvider';
import { toOpenAITools } from './LLMProvider';

interface OpenAIMessage {
	role: 'system' | 'user' | 'assistant' | 'tool';
	content: string | null;
	tool_calls?: Array<{
		id: string;
		type: 'function';
		function: {
			name: string;
			arguments: string;
		};
	}>;
	tool_call_id?: string;
}

interface OpenAIResponse {
	id: string;
	object: string;
	created: number;
	model: string;
	choices: Array<{
		index: number;
		message: {
			role: string;
			content: string | null;
			tool_calls?: Array<{
				id: string;
				type: string;
				function: {
					name: string;
					arguments: string;
				};
			}>;
		};
		finish_reason: string;
	}>;
	usage?: {
		prompt_tokens: number;
		completion_tokens: number;
		total_tokens: number;
	};
	error?: {
		message: string;
		type: string;
		code: string;
	};
}

export class OpenAIProvider implements LLMProvider {
	private apiKey: string;
	private baseUrl: string;

	constructor(apiKey: string, baseUrl: string = 'https://api.openai.com/v1') {
		this.apiKey = apiKey;
		// Remove trailing slash if present
		this.baseUrl = baseUrl.replace(/\/$/, '');
	}

	async chat(
		messages: Message[],
		tools?: ToolDefinition[],
		config?: LLMConfig,
	): Promise<LLMResponse> {
		const model = config?.model || 'gpt-4o-mini';
		const url = `${this.baseUrl}/chat/completions`;

		// Convert messages to OpenAI format
		const openaiMessages = this.convertMessages(messages);

		// Build request body
		const body: Record<string, unknown> = {
			model,
			messages: openaiMessages,
			temperature: config?.temperature ?? 0.7,
			max_tokens: config?.maxTokens ?? 4096,
		};

		if (config?.topP !== undefined) {
			body.top_p = config.topP;
		}

		// Add tools if provided
		if (tools && tools.length > 0) {
			body.tools = toOpenAITools(tools);
			body.tool_choice = 'auto';
		}

		// Make the request
		const response = await fetch(url, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
				'Authorization': `Bearer ${this.apiKey}`,
			},
			body: JSON.stringify(body),
		});

		if (!response.ok) {
			const errorText = await response.text();
			throw new Error(`OpenAI API error: ${response.status} - ${errorText}`);
		}

		const data = (await response.json()) as OpenAIResponse;

		if (data.error) {
			throw new Error(`OpenAI API error: ${data.error.message}`);
		}

		return this.parseResponse(data);
	}

	private convertMessages(messages: Message[]): OpenAIMessage[] {
		const openaiMessages: OpenAIMessage[] = [];

		for (const msg of messages) {
			if (msg.role === 'system') {
				openaiMessages.push({
					role: 'system',
					content: msg.content,
				});
			} else if (msg.role === 'user') {
				openaiMessages.push({
					role: 'user',
					content: msg.content,
				});
			} else if (msg.role === 'assistant') {
				const assistantMsg: OpenAIMessage = {
					role: 'assistant',
					content: msg.content || null,
				};

				if (msg.toolCalls && msg.toolCalls.length > 0) {
					assistantMsg.tool_calls = msg.toolCalls.map((tc) => ({
						id: tc.id,
						type: 'function' as const,
						function: {
							name: tc.name,
							arguments: JSON.stringify(tc.arguments),
						},
					}));
				}

				openaiMessages.push(assistantMsg);
			} else if (msg.role === 'tool') {
				openaiMessages.push({
					role: 'tool',
					content: msg.content,
					tool_call_id: msg.toolCallId,
				});
			}
		}

		return openaiMessages;
	}

	private parseResponse(data: OpenAIResponse): LLMResponse {
		const choice = data.choices?.[0];

		if (!choice) {
			return {
				content: null,
				toolCalls: [],
				finishReason: 'error',
			};
		}

		const toolCalls: ToolCall[] = [];

		if (choice.message.tool_calls) {
			for (const tc of choice.message.tool_calls) {
				let args: Record<string, unknown> = {};
				try {
					args = JSON.parse(tc.function.arguments);
				} catch {
					// If parsing fails, use empty object
					args = {};
				}

				toolCalls.push({
					id: tc.id,
					name: tc.function.name,
					arguments: args,
				});
			}
		}

		// Determine finish reason
		let finishReason: LLMResponse['finishReason'] = 'stop';
		if (choice.finish_reason === 'tool_calls') {
			finishReason = 'tool_calls';
		} else if (choice.finish_reason === 'length') {
			finishReason = 'length';
		}

		return {
			content: choice.message.content,
			toolCalls,
			finishReason,
			usage: data.usage ? {
				promptTokens: data.usage.prompt_tokens,
				completionTokens: data.usage.completion_tokens,
				totalTokens: data.usage.total_tokens,
			} : undefined,
		};
	}
}
