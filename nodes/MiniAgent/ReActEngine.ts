/**
 * ReAct Engine
 * Implements the Reasoning + Acting (ReAct) pattern for AI agents
 * Loop: Think -> Act -> Observe -> Repeat until done
 *
 * Key features:
 * - Maintains full conversation history including tool calls
 * - Configurable max iterations to prevent infinite loops
 * - Proper error handling and recovery
 */

import type { LLMProvider, LLMConfig, Message, ToolCall } from './LLMProvider';
import type { ToolExecutor, ToolResult } from './ToolExecutor';
import type { MemoryManager } from './MemoryManager';
import { createMessage } from './MemoryManager';

export interface ReActConfig {
	maxIterations: number;
	systemMessage?: string;
	verbose?: boolean;
}

export interface ReActResult {
	success: boolean;
	response: string;
	iterations: number;
	toolCalls: Array<{
		tool: string;
		input: Record<string, unknown>;
		output: string;
		success: boolean;
	}>;
	error?: string;
	usage?: {
		promptTokens: number;
		completionTokens: number;
		totalTokens: number;
	};
}

export class ReActEngine {
	private llm: LLMProvider;
	private toolExecutor: ToolExecutor;
	private memory: MemoryManager;
	private config: ReActConfig;
	private llmConfig: LLMConfig;

	constructor(
		llm: LLMProvider,
		toolExecutor: ToolExecutor,
		memory: MemoryManager,
		config: ReActConfig,
		llmConfig: LLMConfig,
	) {
		this.llm = llm;
		this.toolExecutor = toolExecutor;
		this.memory = memory;
		this.config = config;
		this.llmConfig = llmConfig;
	}

	/**
	 * Run the ReAct loop for a user message
	 */
	async run(userMessage: string): Promise<ReActResult> {
		// Load existing conversation history
		const messages: Message[] = await this.memory.load();

		// Add system message if not present and configured
		if (this.config.systemMessage && !messages.find((m) => m.role === 'system')) {
			messages.unshift(createMessage('system', this.config.systemMessage));
		}

		// Add user message
		messages.push(createMessage('user', userMessage));

		// Get tool definitions
		const tools = this.toolExecutor.getDefinitions();

		// Track results
		const toolCallsHistory: ReActResult['toolCalls'] = [];
		let totalUsage = { promptTokens: 0, completionTokens: 0, totalTokens: 0 };
		let iterations = 0;
		let finished = false;
		let finalResponse = '';
		let error: string | undefined;

		// ReAct loop
		while (!finished && iterations < this.config.maxIterations) {
			iterations++;

			try {
				// Call the LLM
				const response = await this.llm.chat(
					messages,
					tools.length > 0 ? tools : undefined,
					this.llmConfig,
				);

				// Track usage
				if (response.usage) {
					totalUsage.promptTokens += response.usage.promptTokens;
					totalUsage.completionTokens += response.usage.completionTokens;
					totalUsage.totalTokens += response.usage.totalTokens;
				}

				// Handle tool calls
				if (response.toolCalls.length > 0) {
					// Add assistant message with tool calls
					messages.push({
						role: 'assistant',
						content: response.content || '',
						toolCalls: response.toolCalls,
					});

					// Execute each tool call
					for (const toolCall of response.toolCalls) {
						const result = await this.executeTool(toolCall);

						// Add tool result to messages
						messages.push(createMessage('tool', result.output || result.error || 'No output', {
							toolCallId: toolCall.id,
						}));

						// Track tool call
						toolCallsHistory.push({
							tool: toolCall.name,
							input: toolCall.arguments,
							output: result.output || result.error || '',
							success: result.success,
						});
					}
				} else {
					// No tool calls - we have a final response
					finished = true;
					finalResponse = response.content || '';

					// Add assistant response to messages
					if (response.content) {
						messages.push(createMessage('assistant', response.content));
					}
				}

				// Check for stop conditions
				if (response.finishReason === 'length') {
					error = 'Response truncated due to token limit';
					finished = true;
				}

			} catch (err) {
				error = err instanceof Error ? err.message : String(err);
				finished = true;
			}
		}

		// Check if we hit max iterations
		if (!finished && iterations >= this.config.maxIterations) {
			error = `Reached maximum iterations (${this.config.maxIterations})`;

			// Add a message indicating we stopped
			messages.push(createMessage('assistant',
				'I apologize, but I was unable to complete the task within the allowed number of steps. ' +
				'Please try breaking down your request into smaller parts.',
			));
		}

		// Save updated conversation history
		await this.memory.save(messages);

		return {
			success: !error,
			response: finalResponse,
			iterations,
			toolCalls: toolCallsHistory,
			error,
			usage: totalUsage.totalTokens > 0 ? totalUsage : undefined,
		};
	}

	/**
	 * Execute a single tool call with error handling
	 */
	private async executeTool(toolCall: ToolCall): Promise<ToolResult> {
		try {
			return await this.toolExecutor.execute(toolCall);
		} catch (err) {
			return {
				success: false,
				output: '',
				error: `Tool execution failed: ${err instanceof Error ? err.message : String(err)}`,
			};
		}
	}
}

/**
 * Create a simple agent for one-shot queries (no memory)
 */
export async function runSimpleAgent(
	llm: LLMProvider,
	toolExecutor: ToolExecutor,
	userMessage: string,
	systemMessage: string,
	llmConfig: LLMConfig,
	maxIterations: number = 10,
): Promise<ReActResult> {
	const messages: Message[] = [
		createMessage('system', systemMessage),
		createMessage('user', userMessage),
	];

	const tools = toolExecutor.getDefinitions();
	const toolCallsHistory: ReActResult['toolCalls'] = [];
	let totalUsage = { promptTokens: 0, completionTokens: 0, totalTokens: 0 };
	let iterations = 0;
	let finished = false;
	let finalResponse = '';
	let error: string | undefined;

	while (!finished && iterations < maxIterations) {
		iterations++;

		try {
			const response = await llm.chat(
				messages,
				tools.length > 0 ? tools : undefined,
				llmConfig,
			);

			if (response.usage) {
				totalUsage.promptTokens += response.usage.promptTokens;
				totalUsage.completionTokens += response.usage.completionTokens;
				totalUsage.totalTokens += response.usage.totalTokens;
			}

			if (response.toolCalls.length > 0) {
				messages.push({
					role: 'assistant',
					content: response.content || '',
					toolCalls: response.toolCalls,
				});

				for (const toolCall of response.toolCalls) {
					const result = await toolExecutor.execute(toolCall);

					messages.push(createMessage('tool', result.output || result.error || 'No output', {
						toolCallId: toolCall.id,
					}));

					toolCallsHistory.push({
						tool: toolCall.name,
						input: toolCall.arguments,
						output: result.output || result.error || '',
						success: result.success,
					});
				}
			} else {
				finished = true;
				finalResponse = response.content || '';
			}

		} catch (err) {
			error = err instanceof Error ? err.message : String(err);
			finished = true;
		}
	}

	if (!finished && iterations >= maxIterations) {
		error = `Reached maximum iterations (${maxIterations})`;
	}

	return {
		success: !error,
		response: finalResponse,
		iterations,
		toolCalls: toolCallsHistory,
		error,
		usage: totalUsage.totalTokens > 0 ? totalUsage : undefined,
	};
}
