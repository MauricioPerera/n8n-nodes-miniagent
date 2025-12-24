import type {
	IExecuteFunctions,
	INodeExecutionData,
	INodeType,
	INodeTypeDescription,
	IDataObject,
} from 'n8n-workflow';

import type { LLMProvider, LLMConfig } from './LLMProvider';
import { GeminiProvider } from './GeminiProvider';
import { AnthropicProvider } from './AnthropicProvider';
import { OpenAIProvider } from './OpenAIProvider';
import { MemoryManager, type MemoryType } from './MemoryManager';
import { ToolExecutor, parseTools, type ExecutableTool } from './ToolExecutor';
import { ReActEngine, runSimpleAgent } from './ReActEngine';

export class MiniAgent implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'Mini Agent',
		name: 'miniAgent',
		icon: 'file:MiniAgent.node.svg',
		group: ['transform'],
		version: 1,
		subtitle: '={{$parameter["operation"]}}',
		description: 'Lightweight AI Agent - zero dependencies, built-in memory, multi-LLM support',
		defaults: {
			name: 'Mini Agent',
		},
		inputs: ['main'],
		outputs: ['main'],
		credentials: [
			{
				name: 'geminiApi',
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
				name: 'openAiCompatibleApi',
				required: true,
				displayOptions: {
					show: {
						provider: ['openai-compatible'],
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
						name: 'OpenAI Compatible',
						value: 'openai-compatible',
						description: 'OpenAI, OpenRouter, Groq, Ollama, LM Studio, etc.',
					},
				],
				default: 'gemini',
				description: 'The LLM provider to use',
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
						operation: ['chat', 'chatWithMemory'],
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
						operation: ['chat', 'chatWithMemory'],
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
						operation: ['chatWithMemory', 'clearMemory', 'getMemory'],
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
						operation: ['chat', 'chatWithMemory'],
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

async function createLLMProvider(ctx: IExecuteFunctions, itemIndex: number): Promise<LLMProvider> {
	const provider = ctx.getNodeParameter('provider', itemIndex) as string;

	if (provider === 'gemini') {
		const credentials = await ctx.getCredentials('geminiApi', itemIndex);
		return new GeminiProvider(credentials.apiKey as string);
	}

	if (provider === 'anthropic') {
		const credentials = await ctx.getCredentials('anthropicApi', itemIndex);
		return new AnthropicProvider(credentials.apiKey as string);
	}

	if (provider === 'openai-compatible') {
		const credentials = await ctx.getCredentials('openAiCompatibleApi', itemIndex);
		return new OpenAIProvider(
			credentials.apiKey as string,
			credentials.baseUrl as string,
		);
	}

	throw new Error(`Unknown provider: ${provider}`);
}
