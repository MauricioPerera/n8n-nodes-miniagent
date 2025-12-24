/**
 * Anthropic API Provider
 * Uses Claude's Messages API directly (no SDK)
 * Supports tool_use for function calling
 */

import type {
	LLMProvider,
	LLMResponse,
	LLMConfig,
	Message,
	ToolDefinition,
	ToolCall,
} from './LLMProvider';
import { toAnthropicTools } from './LLMProvider';

const ANTHROPIC_API_BASE = 'https://api.anthropic.com/v1';
const ANTHROPIC_VERSION = '2023-06-01';

interface AnthropicMessage {
	role: 'user' | 'assistant';
	content: string | AnthropicContent[];
}

interface AnthropicContent {
	type: 'text' | 'tool_use' | 'tool_result';
	text?: string;
	id?: string;
	name?: string;
	input?: Record<string, unknown>;
	tool_use_id?: string;
	content?: string;
}

interface AnthropicResponse {
	id: string;
	type: string;
	role: string;
	content: AnthropicContent[];
	model: string;
	stop_reason: string | null;
	stop_sequence: string | null;
	usage: {
		input_tokens: number;
		output_tokens: number;
	};
	error?: {
		type: string;
		message: string;
	};
}

export class AnthropicProvider implements LLMProvider {
	private apiKey: string;

	constructor(apiKey: string) {
		this.apiKey = apiKey;
	}

	async chat(
		messages: Message[],
		tools?: ToolDefinition[],
		config?: LLMConfig,
	): Promise<LLMResponse> {
		const model = config?.model || 'claude-3-5-sonnet-20241022';
		const url = `${ANTHROPIC_API_BASE}/messages`;

		// Convert messages to Anthropic format
		const { system, anthropicMessages } = this.convertMessages(messages);

		// Build request body
		const body: Record<string, unknown> = {
			model,
			max_tokens: config?.maxTokens ?? 4096,
			messages: anthropicMessages,
		};

		if (system) {
			body.system = system;
		}

		if (config?.temperature !== undefined) {
			body.temperature = config.temperature;
		}

		if (config?.topP !== undefined) {
			body.top_p = config.topP;
		}

		// Add tools if provided
		if (tools && tools.length > 0) {
			body.tools = toAnthropicTools(tools);
		}

		// Make the request
		const response = await fetch(url, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
				'x-api-key': this.apiKey,
				'anthropic-version': ANTHROPIC_VERSION,
			},
			body: JSON.stringify(body),
		});

		if (!response.ok) {
			const errorText = await response.text();
			throw new Error(`Anthropic API error: ${response.status} - ${errorText}`);
		}

		const data = (await response.json()) as AnthropicResponse;

		if (data.error) {
			throw new Error(`Anthropic API error: ${data.error.message}`);
		}

		return this.parseResponse(data);
	}

	private convertMessages(messages: Message[]): {
		system: string | null;
		anthropicMessages: AnthropicMessage[];
	} {
		let system: string | null = null;
		const anthropicMessages: AnthropicMessage[] = [];

		for (const msg of messages) {
			if (msg.role === 'system') {
				system = msg.content;
				continue;
			}

			if (msg.role === 'user') {
				anthropicMessages.push({
					role: 'user',
					content: msg.content,
				});
			} else if (msg.role === 'assistant') {
				const content: AnthropicContent[] = [];

				if (msg.content) {
					content.push({
						type: 'text',
						text: msg.content,
					});
				}

				if (msg.toolCalls) {
					for (const tc of msg.toolCalls) {
						content.push({
							type: 'tool_use',
							id: tc.id,
							name: tc.name,
							input: tc.arguments,
						});
					}
				}

				anthropicMessages.push({
					role: 'assistant',
					content: content.length === 1 && content[0].type === 'text'
						? content[0].text!
						: content,
				});
			} else if (msg.role === 'tool') {
				// Tool results go in a user message
				anthropicMessages.push({
					role: 'user',
					content: [{
						type: 'tool_result',
						tool_use_id: msg.toolCallId || '',
						content: msg.content,
					}],
				});
			}
		}

		return { system, anthropicMessages };
	}

	private parseResponse(data: AnthropicResponse): LLMResponse {
		const toolCalls: ToolCall[] = [];
		let content: string | null = null;

		for (const block of data.content) {
			if (block.type === 'text' && block.text) {
				content = (content || '') + block.text;
			}
			if (block.type === 'tool_use' && block.id && block.name) {
				toolCalls.push({
					id: block.id,
					name: block.name,
					arguments: block.input || {},
				});
			}
		}

		// Determine finish reason
		let finishReason: LLMResponse['finishReason'] = 'stop';
		if (toolCalls.length > 0 || data.stop_reason === 'tool_use') {
			finishReason = 'tool_calls';
		} else if (data.stop_reason === 'max_tokens') {
			finishReason = 'length';
		}

		return {
			content,
			toolCalls,
			finishReason,
			usage: {
				promptTokens: data.usage.input_tokens,
				completionTokens: data.usage.output_tokens,
				totalTokens: data.usage.input_tokens + data.usage.output_tokens,
			},
		};
	}
}
