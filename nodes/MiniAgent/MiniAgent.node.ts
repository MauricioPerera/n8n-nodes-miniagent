import type {
	IExecuteFunctions,
	INodeExecutionData,
	INodeType,
	INodeTypeDescription,
	IDataObject,
} from 'n8n-workflow';

import type { LLMProvider, LLMConfig, Message } from './LLMProvider';
import { GeminiProvider } from './GeminiProvider';
import { AnthropicProvider } from './AnthropicProvider';
import { OpenAIProvider } from './OpenAIProvider';
import { MemoryManager, type MemoryType } from './MemoryManager';
import { ToolExecutor, parseTools, type ExecutableTool } from './ToolExecutor';
import { ReActEngine, runSimpleAgent } from './ReActEngine';
import { RAGMemory, getEmbeddingDimensions } from './RAGMemory';
import type { SearchMode } from './HybridSearch';

export class MiniAgent implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'Mini Agent',
		name: 'miniAgent',
		icon: 'file:MiniAgent.node.svg',
		group: ['transform'],
		version: 1,
		usableAsTool: true,
		subtitle: '={{$parameter["operation"]}}',
		description: 'Lightweight AI Agent - zero dependencies, built-in memory, multi-LLM support',
		defaults: {
			name: 'Mini Agent',
		},
		inputs: ['main'],
		outputs: ['main'],
		credentials: [
			{
				name: 'googlePalmApi',
				required: true,
				displayOptions: {
					show: {
						provider: ['gemini'],
					},
				},
			},
			{
				name: 'anthropicApi',
				required: true,
				displayOptions: {
					show: {
						provider: ['anthropic'],
					},
				},
			},
			{
				name: 'openAiApi',
				required: true,
				displayOptions: {
					show: {
						provider: ['openai'],
					},
				},
			},
		],
		properties: [
			// Operation selector
			{
				displayName: 'Operation',
				name: 'operation',
				type: 'options',
				noDataExpression: true,
				options: [
					{
						name: 'Chat',
						value: 'chat',
						description: 'Send a message and get a response (no memory)',
						action: 'Send a chat message',
					},
					{
						name: 'Chat with Memory',
						value: 'chatWithMemory',
						description: 'Chat with conversation history preserved',
						action: 'Chat with memory',
					},
					{
						name: 'Chat with RAG',
						value: 'chatWithRAG',
						description: 'Chat with semantic search over conversation history (requires embeddings)',
						action: 'Chat with RAG memory',
					},
					{
						name: 'Clear Memory',
						value: 'clearMemory',
						description: 'Clear conversation history for a session',
						action: 'Clear memory',
					},
					{
						name: 'Get Memory',
						value: 'getMemory',
						description: 'Get current conversation history',
						action: 'Get memory',
					},
				],
				default: 'chat',
			},

			// === LLM Provider Settings ===
			{
				displayName: 'Provider',
				name: 'provider',
				type: 'options',
				options: [
					{
						name: 'Gemini',
						value: 'gemini',
						description: 'Google Gemini (gemini-pro, gemini-1.5-flash, etc.)',
					},
					{
						name: 'Anthropic (Claude)',
						value: 'anthropic',
						description: 'Claude models (claude-3-opus, claude-3-sonnet, etc.)',
					},
					{
						name: 'OpenAI',
						value: 'openai',
						description: 'OpenAI GPT models (gpt-4o, gpt-4o-mini, etc.)',
					},
					{
						name: 'OpenAI Compatible (Custom)',
						value: 'openai-compatible',
						description: 'OpenRouter, Groq, Ollama, LM Studio, or any OpenAI-compatible API',
					},
				],
				default: 'gemini',
				description: 'The LLM provider to use',
			},
			// === Custom OpenAI-Compatible Settings ===
			{
				displayName: 'API Key',
				name: 'customApiKey',
				type: 'string',
				typeOptions: {
					password: true,
				},
				default: '',
				description: 'API key for the OpenAI-compatible service',
				displayOptions: {
					show: {
						provider: ['openai-compatible'],
					},
				},
			},
			{
				displayName: 'Base URL',
				name: 'customBaseUrl',
				type: 'string',
				default: 'https://api.openai.com/v1',
				placeholder: 'https://api.groq.com/openai/v1',
				description: 'Base URL for the OpenAI-compatible API. Examples: https://api.groq.com/openai/v1, https://openrouter.ai/api/v1, http://localhost:11434/v1',
				displayOptions: {
					show: {
						provider: ['openai-compatible'],
					},
				},
			},
			{
				displayName: 'Model',
				name: 'model',
				type: 'string',
				default: '',
				placeholder: 'gemini-1.5-flash',
				description: 'The model to use. Examples: gemini-1.5-flash, claude-3-5-sonnet-20241022, gpt-4o-mini',
				displayOptions: {
					hide: {
						operation: ['clearMemory', 'getMemory'],
					},
				},
			},

			// === Message Input ===
			{
				displayName: 'Message',
				name: 'message',
				type: 'string',
				typeOptions: {
					rows: 4,
				},
				default: '',
				placeholder: 'Enter your message...',
				description: 'The message to send to the AI agent',
				displayOptions: {
					show: {
						operation: ['chat', 'chatWithMemory', 'chatWithRAG'],
					},
				},
			},

			// === System Prompt ===
			{
				displayName: 'System Message',
				name: 'systemMessage',
				type: 'string',
				typeOptions: {
					rows: 4,
				},
				default: 'You are a helpful AI assistant.',
				description: 'The system prompt that defines the agent\'s behavior',
				displayOptions: {
					show: {
						operation: ['chat', 'chatWithMemory', 'chatWithRAG'],
					},
				},
			},

			// === Memory Settings ===
			{
				displayName: 'Session ID',
				name: 'sessionId',
				type: 'string',
				default: '={{ $json.sessionId || "default" }}',
				description: 'Unique identifier for the conversation session',
				displayOptions: {
					show: {
						operation: ['chatWithMemory', 'chatWithRAG', 'clearMemory', 'getMemory'],
					},
				},
			},
			{
				displayName: 'Memory Type',
				name: 'memoryType',
				type: 'options',
				options: [
					{
						name: 'Buffer (Volatile)',
						value: 'buffer',
						description: 'In-memory storage, lost when n8n restarts',
					},
					{
						name: 'Workflow Static Data (Persistent)',
						value: 'static-data',
						description: 'Persists across n8n restarts in workflow data',
					},
				],
				default: 'buffer',
				description: 'How to store conversation history',
				displayOptions: {
					show: {
						operation: ['chatWithMemory', 'clearMemory', 'getMemory'],
					},
				},
			},

			// === RAG Settings ===
			{
				displayName: 'Search Mode',
				name: 'ragSearchMode',
				type: 'options',
				options: [
					{
						name: 'Hybrid (Vector + Keyword)',
						value: 'hybrid',
						description: 'Combines semantic and keyword search for best results',
					},
					{
						name: 'Vector Only',
						value: 'vector',
						description: 'Pure semantic similarity search',
					},
					{
						name: 'Keyword Only',
						value: 'keyword',
						description: 'BM25 keyword search only',
					},
				],
				default: 'hybrid',
				description: 'How to search the conversation history',
				displayOptions: {
					show: {
						operation: ['chatWithRAG'],
					},
				},
			},
			{
				displayName: 'Max Context Messages',
				name: 'ragMaxContext',
				type: 'number',
				typeOptions: {
					minValue: 1,
					maxValue: 50,
				},
				default: 5,
				description: 'Maximum number of relevant past messages to include as context',
				displayOptions: {
					show: {
						operation: ['chatWithRAG'],
					},
				},
			},
			{
				displayName: 'Min Similarity',
				name: 'ragMinSimilarity',
				type: 'number',
				typeOptions: {
					minValue: 0,
					maxValue: 1,
					numberPrecision: 2,
				},
				default: 0.3,
				description: 'Minimum similarity score (0-1) for context messages',
				displayOptions: {
					show: {
						operation: ['chatWithRAG'],
					},
				},
			},
			{
				displayName: 'Persist RAG Store',
				name: 'ragPersist',
				type: 'boolean',
				default: false,
				description: 'Whether to persist the RAG memory to workflow static data',
				displayOptions: {
					show: {
						operation: ['chatWithRAG'],
					},
				},
			},

			// === Tools Configuration ===
			{
				displayName: 'Tools',
				name: 'tools',
				type: 'json',
				default: '[]',
				placeholder: 'Add tools as JSON array...',
				description: 'Tools the agent can use. See documentation for format.',
				typeOptions: {
					rows: 10,
				},
				displayOptions: {
					show: {
						operation: ['chat', 'chatWithMemory', 'chatWithRAG'],
					},
				},
			},

			// === Advanced Options ===
			{
				displayName: 'Options',
				name: 'options',
				type: 'collection',
				placeholder: 'Add Option',
				default: {},
				options: [
					{
						displayName: 'Temperature',
						name: 'temperature',
						type: 'number',
						typeOptions: {
							minValue: 0,
							maxValue: 2,
							numberPrecision: 1,
						},
						default: 0.7,
						description: 'Controls randomness. Lower = more focused, higher = more creative.',
					},
					{
						displayName: 'Max Tokens',
						name: 'maxTokens',
						type: 'number',
						typeOptions: {
							minValue: 1,
							maxValue: 128000,
						},
						default: 4096,
						description: 'Maximum number of tokens in the response',
					},
					{
						displayName: 'Max Iterations',
						name: 'maxIterations',
						type: 'number',
						typeOptions: {
							minValue: 1,
							maxValue: 50,
						},
						default: 10,
						description: 'Maximum number of tool-use loops before stopping',
					},
					{
						displayName: 'Max Memory Messages',
						name: 'maxMemoryMessages',
						type: 'number',
						typeOptions: {
							minValue: 1,
							maxValue: 1000,
						},
						default: 50,
						description: 'Maximum messages to keep in memory',
					},
					{
						displayName: 'Include Tool Calls in Memory',
						name: 'includeToolCalls',
						type: 'boolean',
						default: true,
						description: 'Whether to save tool calls and results in memory. Recommended: true (fixes the issue where agent stops using tools).',
					},
				],
				displayOptions: {
					hide: {
						operation: ['clearMemory', 'getMemory'],
					},
				},
			},
		],
	};

	async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
		const items = this.getInputData();
		const returnData: INodeExecutionData[] = [];

		for (let i = 0; i < items.length; i++) {
			try {
				const operation = this.getNodeParameter('operation', i) as string;

				if (operation === 'clearMemory') {
					const result = await handleClearMemory(this, i);
					returnData.push({ json: result });
				} else if (operation === 'getMemory') {
					const result = await handleGetMemory(this, i);
					returnData.push({ json: result });
				} else if (operation === 'chat') {
					const result = await handleChat(this, i);
					returnData.push({ json: result });
				} else if (operation === 'chatWithMemory') {
					const result = await handleChatWithMemory(this, i);
					returnData.push({ json: result });
				} else if (operation === 'chatWithRAG') {
					const result = await handleChatWithRAG(this, i);
					returnData.push({ json: result });
				}
			} catch (error) {
				if (this.continueOnFail()) {
					returnData.push({
						json: {
							error: error instanceof Error ? error.message : String(error),
						},
					});
					continue;
				}
				throw error;
			}
		}

		return [returnData];
	}
}

// Helper functions

async function handleClearMemory(ctx: IExecuteFunctions, itemIndex: number): Promise<IDataObject> {
	const sessionId = ctx.getNodeParameter('sessionId', itemIndex) as string;
	const memoryType = ctx.getNodeParameter('memoryType', itemIndex) as MemoryType;

	const memory = new MemoryManager(
		{ type: memoryType, sessionId },
		ctx,
	);

	await memory.clear();

	return {
		success: true,
		sessionId,
		message: 'Memory cleared successfully',
	};
}

async function handleGetMemory(ctx: IExecuteFunctions, itemIndex: number): Promise<IDataObject> {
	const sessionId = ctx.getNodeParameter('sessionId', itemIndex) as string;
	const memoryType = ctx.getNodeParameter('memoryType', itemIndex) as MemoryType;

	const memory = new MemoryManager(
		{ type: memoryType, sessionId },
		ctx,
	);

	const messages = await memory.getAll();

	return {
		sessionId,
		messageCount: messages.length,
		messages,
	};
}

async function handleChat(ctx: IExecuteFunctions, itemIndex: number): Promise<IDataObject> {
	const message = ctx.getNodeParameter('message', itemIndex) as string;
	const systemMessage = ctx.getNodeParameter('systemMessage', itemIndex) as string;
	const toolsJson = ctx.getNodeParameter('tools', itemIndex) as string;
	const options = ctx.getNodeParameter('options', itemIndex) as {
		temperature?: number;
		maxTokens?: number;
		maxIterations?: number;
	};

	// Create LLM provider
	const llm = await createLLMProvider(ctx, itemIndex);

	// Create LLM config
	const llmConfig: LLMConfig = {
		model: ctx.getNodeParameter('model', itemIndex) as string,
		temperature: options.temperature ?? 0.7,
		maxTokens: options.maxTokens ?? 4096,
	};

	// Parse and create tool executor
	const tools = parseTools(toolsJson) as ExecutableTool[];
	const toolExecutor = new ToolExecutor(tools);

	// Run simple agent (no memory)
	const result = await runSimpleAgent(
		llm,
		toolExecutor,
		message,
		systemMessage,
		llmConfig,
		options.maxIterations ?? 10,
	);

	return {
		success: result.success,
		response: result.response,
		iterations: result.iterations,
		toolCalls: result.toolCalls,
		...(result.error && { error: result.error }),
		...(result.usage && { usage: result.usage }),
	};
}

async function handleChatWithMemory(ctx: IExecuteFunctions, itemIndex: number): Promise<IDataObject> {
	const message = ctx.getNodeParameter('message', itemIndex) as string;
	const systemMessage = ctx.getNodeParameter('systemMessage', itemIndex) as string;
	const sessionId = ctx.getNodeParameter('sessionId', itemIndex) as string;
	const memoryType = ctx.getNodeParameter('memoryType', itemIndex) as MemoryType;
	const toolsJson = ctx.getNodeParameter('tools', itemIndex) as string;
	const options = ctx.getNodeParameter('options', itemIndex) as {
		temperature?: number;
		maxTokens?: number;
		maxIterations?: number;
		maxMemoryMessages?: number;
		includeToolCalls?: boolean;
	};

	// Create LLM provider
	const llm = await createLLMProvider(ctx, itemIndex);

	// Create LLM config
	const llmConfig: LLMConfig = {
		model: ctx.getNodeParameter('model', itemIndex) as string,
		temperature: options.temperature ?? 0.7,
		maxTokens: options.maxTokens ?? 4096,
	};

	// Create memory manager
	const memory = new MemoryManager(
		{
			type: memoryType,
			sessionId,
			maxMessages: options.maxMemoryMessages ?? 50,
			includeToolCalls: options.includeToolCalls ?? true,
		},
		ctx,
	);

	// Parse and create tool executor
	const tools = parseTools(toolsJson) as ExecutableTool[];
	const toolExecutor = new ToolExecutor(tools);

	// Create and run ReAct engine
	const engine = new ReActEngine(
		llm,
		toolExecutor,
		memory,
		{
			maxIterations: options.maxIterations ?? 10,
			systemMessage,
		},
		llmConfig,
	);

	const result = await engine.run(message);

	return {
		success: result.success,
		response: result.response,
		sessionId,
		iterations: result.iterations,
		toolCalls: result.toolCalls,
		...(result.error && { error: result.error }),
		...(result.usage && { usage: result.usage }),
	};
}

async function handleChatWithRAG(ctx: IExecuteFunctions, itemIndex: number): Promise<IDataObject> {
	const message = ctx.getNodeParameter('message', itemIndex) as string;
	const systemMessage = ctx.getNodeParameter('systemMessage', itemIndex) as string;
	const sessionId = ctx.getNodeParameter('sessionId', itemIndex) as string;
	const toolsJson = ctx.getNodeParameter('tools', itemIndex) as string;
	const searchMode = ctx.getNodeParameter('ragSearchMode', itemIndex) as SearchMode;
	const maxContext = ctx.getNodeParameter('ragMaxContext', itemIndex) as number;
	const minSimilarity = ctx.getNodeParameter('ragMinSimilarity', itemIndex) as number;
	const persistRAG = ctx.getNodeParameter('ragPersist', itemIndex) as boolean;
	const options = ctx.getNodeParameter('options', itemIndex) as {
		temperature?: number;
		maxTokens?: number;
		maxIterations?: number;
	};

	// Create LLM provider
	const llm = await createLLMProvider(ctx, itemIndex);
	const provider = ctx.getNodeParameter('provider', itemIndex) as string;

	// Check if provider supports embeddings
	if (!llm.supportsEmbeddings || !llm.supportsEmbeddings()) {
		throw new Error(`Provider "${provider}" does not support embeddings. Use Gemini or OpenAI for RAG memory.`);
	}

	// Create LLM config
	const llmConfig: LLMConfig = {
		model: ctx.getNodeParameter('model', itemIndex) as string,
		temperature: options.temperature ?? 0.7,
		maxTokens: options.maxTokens ?? 4096,
	};

	// Get embedding dimensions based on provider
	const dimensions = getEmbeddingDimensions(provider, llmConfig.model);

	// Create RAG memory
	const ragMemory = new RAGMemory(
		{
			sessionId,
			dimensions,
			maxContextMessages: maxContext,
			minSimilarity,
			searchMode,
			persistToStaticData: persistRAG,
		},
		llm,
		ctx,
	);

	// Parse and create tool executor
	const tools = parseTools(toolsJson) as ExecutableTool[];
	const toolExecutor = new ToolExecutor(tools);

	// Get relevant context from RAG memory
	const contextMessages = await ragMemory.buildContextMessages(message);

	// Build conversation with context
	const messages: Message[] = [
		{ role: 'system', content: systemMessage },
		...contextMessages,
		{ role: 'user', content: message },
	];

	// Run agent loop
	let iterations = 0;
	const maxIterations = options.maxIterations ?? 10;
	const toolCallsLog: Array<{ name: string; arguments: Record<string, unknown>; result: string }> = [];
	let response = '';
	let totalUsage = { promptTokens: 0, completionTokens: 0, totalTokens: 0 };

	while (iterations < maxIterations) {
		iterations++;

		// Get tool definitions
		const toolDefs = toolExecutor.getDefinitions();

		// Call LLM
		const llmResponse = await llm.chat(messages, toolDefs.length > 0 ? toolDefs : undefined, llmConfig);

		// Accumulate usage
		if (llmResponse.usage) {
			totalUsage.promptTokens += llmResponse.usage.promptTokens;
			totalUsage.completionTokens += llmResponse.usage.completionTokens;
			totalUsage.totalTokens += llmResponse.usage.totalTokens;
		}

		// If there are tool calls, execute them
		if (llmResponse.toolCalls && llmResponse.toolCalls.length > 0) {
			// Add assistant message with tool calls
			messages.push({
				role: 'assistant',
				content: llmResponse.content || '',
				toolCalls: llmResponse.toolCalls,
			});

			// Execute each tool
			for (const tc of llmResponse.toolCalls) {
				const result = await toolExecutor.execute(tc);
				toolCallsLog.push({
					name: tc.name,
					arguments: tc.arguments,
					result: result.output,
				});

				// Add tool result
				messages.push({
					role: 'tool',
					content: result.output,
					toolCallId: tc.id,
				});
			}

			continue;
		}

		// Final response
		response = llmResponse.content || '';
		break;
	}

	// Save user message and response to RAG memory
	await ragMemory.addMessage({ role: 'user', content: message });
	await ragMemory.addMessage({ role: 'assistant', content: response });

	// Get RAG stats
	const ragStats = ragMemory.getStats();

	return {
		success: true,
		response,
		sessionId,
		iterations,
		toolCalls: toolCallsLog,
		ragStats: {
			messageCount: ragStats.messageCount,
			documentCount: ragStats.documentCount,
		},
		usage: totalUsage,
	};
}

async function createLLMProvider(ctx: IExecuteFunctions, itemIndex: number): Promise<LLMProvider> {
	const provider = ctx.getNodeParameter('provider', itemIndex) as string;

	if (provider === 'gemini') {
		const credentials = await ctx.getCredentials('googlePalmApi', itemIndex);
		return new GeminiProvider(credentials.apiKey as string);
	}

	if (provider === 'anthropic') {
		const credentials = await ctx.getCredentials('anthropicApi', itemIndex);
		return new AnthropicProvider(credentials.apiKey as string);
	}

	if (provider === 'openai') {
		const credentials = await ctx.getCredentials('openAiApi', itemIndex);
		return new OpenAIProvider(
			credentials.apiKey as string,
			'https://api.openai.com/v1',
		);
	}

	if (provider === 'openai-compatible') {
		const apiKey = ctx.getNodeParameter('customApiKey', itemIndex) as string;
		const baseUrl = ctx.getNodeParameter('customBaseUrl', itemIndex) as string;
		return new OpenAIProvider(apiKey, baseUrl);
	}

	throw new Error(`Unknown provider: ${provider}`);
}
