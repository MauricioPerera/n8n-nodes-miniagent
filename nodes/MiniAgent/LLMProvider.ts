/**
 * Common types and interfaces for LLM providers
 * All providers implement the same interface for easy swapping
 * Supports both chat and embeddings
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

// Configuration for embedding requests
export interface EmbeddingConfig {
	model?: string;
	dimensions?: number;
}

// Common interface for all LLM providers
export interface LLMProvider {
	/**
	 * Send a chat request to the LLM
	 */
	chat(
		messages: Message[],
		tools?: ToolDefinition[],
		config?: LLMConfig,
	): Promise<LLMResponse>;

	/**
	 * Generate embeddings for text (optional - not all providers support this)
	 */
	embed?(
		texts: string[],
		config?: EmbeddingConfig,
	): Promise<number[][]>;

	/**
	 * Check if this provider supports embeddings
	 */
	supportsEmbeddings?(): boolean;
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

// Normalize a vector to unit length
export function normalizeVector(vector: number[]): number[] {
	const magnitude = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
	if (magnitude === 0) return vector;
	return vector.map((val) => val / magnitude);
}

// Calculate cosine similarity between two vectors
export function cosineSimilarity(a: number[], b: number[]): number {
	if (a.length !== b.length) return 0;
	let dotProduct = 0;
	let normA = 0;
	let normB = 0;
	for (let i = 0; i < a.length; i++) {
		dotProduct += a[i] * b[i];
		normA += a[i] * a[i];
		normB += b[i] * b[i];
	}
	const denominator = Math.sqrt(normA) * Math.sqrt(normB);
	return denominator === 0 ? 0 : dotProduct / denominator;
}
