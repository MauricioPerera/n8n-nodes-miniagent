/**
 * Memory Manager
 * Handles conversation history storage with two modes:
 * 1. Buffer (in-memory, volatile)
 * 2. Workflow Static Data (persistent across executions)
 *
 * Key feature: Stores ALL messages including tool calls and results
 * This solves the n8n AI Agent issue #14361 where tool calls were not saved
 */

import type { Message } from './LLMProvider';
import type { IExecuteFunctions } from 'n8n-workflow';

export type MemoryType = 'buffer' | 'static-data';

export interface MemoryConfig {
	type: MemoryType;
	sessionId: string;
	maxMessages?: number;
	includeToolCalls?: boolean;
}

// In-memory buffer storage (singleton pattern like n8n's Simple Memory)
const memoryBuffer: Map<string, {
	messages: Message[];
	lastAccess: number;
}> = new Map();

// TTL for buffer entries (remove if unused for 1 hour)
const BUFFER_TTL = 60 * 60 * 1000; // 1 hour

/**
 * Lazy cleanup of expired buffers (called on access instead of using timers)
 * n8n Cloud doesn't allow setInterval, so we clean up on each access
 */
function cleanupExpiredBuffers(): void {
	const now = Date.now();
	for (const [key, value] of memoryBuffer.entries()) {
		if (now - value.lastAccess > BUFFER_TTL) {
			memoryBuffer.delete(key);
		}
	}
}

export class MemoryManager {
	private config: MemoryConfig;
	private executeFunctions?: IExecuteFunctions;

	constructor(config: MemoryConfig, executeFunctions?: IExecuteFunctions) {
		this.config = config;
		this.executeFunctions = executeFunctions;
	}

	/**
	 * Generate a unique key for this session
	 */
	private getKey(): string {
		const workflowId = this.executeFunctions?.getWorkflow().id ?? 'default';
		return `${workflowId}__${this.config.sessionId}`;
	}

	/**
	 * Load messages from storage
	 */
	async load(): Promise<Message[]> {
		if (this.config.type === 'buffer') {
			return this.loadFromBuffer();
		} else {
			return this.loadFromStaticData();
		}
	}

	/**
	 * Save messages to storage
	 */
	async save(messages: Message[]): Promise<void> {
		// Filter messages based on config
		let messagesToSave = messages;

		if (!this.config.includeToolCalls) {
			messagesToSave = messages.filter(
				(m) => m.role !== 'tool' && !m.toolCalls?.length,
			);
		}

		// Apply max messages limit
		if (this.config.maxMessages && messagesToSave.length > this.config.maxMessages) {
			// Keep system message if present, then most recent messages
			const systemMsg = messagesToSave.find((m) => m.role === 'system');
			const otherMsgs = messagesToSave.filter((m) => m.role !== 'system');
			const recentMsgs = otherMsgs.slice(-this.config.maxMessages + (systemMsg ? 1 : 0));
			messagesToSave = systemMsg ? [systemMsg, ...recentMsgs] : recentMsgs;
		}

		if (this.config.type === 'buffer') {
			this.saveToBuffer(messagesToSave);
		} else {
			await this.saveToStaticData(messagesToSave);
		}
	}

	/**
	 * Clear all messages for this session
	 */
	async clear(): Promise<void> {
		if (this.config.type === 'buffer') {
			memoryBuffer.delete(this.getKey());
		} else {
			await this.saveToStaticData([]);
		}
	}

	/**
	 * Get all messages including tool calls (for debugging)
	 */
	async getAll(): Promise<Message[]> {
		return this.load();
	}

	// Buffer storage methods
	private loadFromBuffer(): Message[] {
		// Lazy cleanup of expired buffers on access
		cleanupExpiredBuffers();

		const key = this.getKey();
		const entry = memoryBuffer.get(key);

		if (entry) {
			entry.lastAccess = Date.now();
			return [...entry.messages];
		}

		return [];
	}

	private saveToBuffer(messages: Message[]): void {
		const key = this.getKey();
		memoryBuffer.set(key, {
			messages: [...messages],
			lastAccess: Date.now(),
		});
	}

	// Static data storage methods
	private loadFromStaticData(): Message[] {
		if (!this.executeFunctions) {
			return [];
		}

		const staticData = this.executeFunctions.getWorkflowStaticData('global');
		const key = `miniagent_${this.config.sessionId}`;
		const stored = staticData[key];

		if (typeof stored === 'string') {
			try {
				return JSON.parse(stored) as Message[];
			} catch {
				return [];
			}
		}

		return [];
	}

	private async saveToStaticData(messages: Message[]): Promise<void> {
		if (!this.executeFunctions) {
			return;
		}

		const staticData = this.executeFunctions.getWorkflowStaticData('global');
		const key = `miniagent_${this.config.sessionId}`;
		staticData[key] = JSON.stringify(messages);
	}
}

/**
 * Serialize a message for storage (handles circular references)
 */
export function serializeMessage(msg: Message): Message {
	return {
		role: msg.role,
		content: msg.content,
		...(msg.toolCallId && { toolCallId: msg.toolCallId }),
		...(msg.toolCalls && { toolCalls: msg.toolCalls }),
	};
}

/**
 * Create a properly formatted message
 */
export function createMessage(
	role: Message['role'],
	content: string,
	options?: {
		toolCallId?: string;
		toolCalls?: Message['toolCalls'];
	},
): Message {
	return {
		role,
		content,
		...options,
	};
}
