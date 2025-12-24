/**
 * Common types and interfaces for LLM providers
 * All providers implement the same interface for easy swapping
 */

// Message types for conversation history
export interface Message {
	role: 'system' | 'user' | 'assistant' | 'tool';
	content: string;
	toolCallId?: string;
	toolCalls?: ToolCall[];
}

// Tool call from the LLM
export interface ToolCall {
	id: string;
	name: string;
	arguments: Record<string, unknown>;
}

// Tool definition for the LLM
export interface ToolDefinition {
	name: string;
	description: string;
	parameters: {
		type: 'object';
		properties: Record<string, {
			type: string;
			description?: string;
			enum?: string[];
		}>;
		required?: string[];
	};
}

// Response from the LLM
export interface LLMResponse {
	content: string | null;
	toolCalls: ToolCall[];
	finishReason: 'stop' | 'tool_calls' | 'length' | 'error';
	usage?: {
		promptTokens: number;
		completionTokens: number;
		totalTokens: number;
	};
}

// Configuration for LLM requests
export interface LLMConfig {
	model: string;
	temperature?: number;
	maxTokens?: number;
	topP?: number;
}

// Common interface for all LLM providers
export interface LLMProvider {
	/**
	 * Send a chat request to the LLM
	 * @param messages - Conversation history
	 * @param tools - Available tools for the LLM to call
	 * @param config - Model configuration
	 * @returns LLM response with content and/or tool calls
	 */
	chat(
		messages: Message[],
		tools?: ToolDefinition[],
		config?: LLMConfig,
	): Promise<LLMResponse>;
}

// Helper to generate unique tool call IDs
export function generateToolCallId(): string {
	return `call_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
}

// Helper to convert tool definitions to OpenAI format
export function toOpenAITools(tools: ToolDefinition[]): object[] {
	return tools.map((tool) => ({
		type: 'function',
		function: {
			name: tool.name,
			description: tool.description,
			parameters: tool.parameters,
		},
	}));
}

// Helper to convert tool definitions to Gemini format
export function toGeminiTools(tools: ToolDefinition[]): object[] {
	return [{
		functionDeclarations: tools.map((tool) => ({
			name: tool.name,
			description: tool.description,
			parameters: tool.parameters,
		})),
	}];
}

// Helper to convert tool definitions to Anthropic format
export function toAnthropicTools(tools: ToolDefinition[]): object[] {
	return tools.map((tool) => ({
		name: tool.name,
		description: tool.description,
		input_schema: tool.parameters,
	}));
}
