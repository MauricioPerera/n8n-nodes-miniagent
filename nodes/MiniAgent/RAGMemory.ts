/**
 * RAG Memory Manager
 * Combines conversation memory with vector search for context retrieval
 *
 * Features:
 * - Stores conversation messages with embeddings
 * - Retrieves relevant context using semantic similarity
 * - Supports hybrid search (vector + keyword)
 * - Persists to workflow static data
 */

import type { Message } from './LLMProvider';
import type { LLMProvider } from './LLMProvider';
import type { IExecuteFunctions } from 'n8n-workflow';
import { VectorStore, type VectorDocument, type SearchResult } from './VectorStore';
import type { HybridSearchResult, SearchMode } from './HybridSearch';

export interface RAGConfig {
	sessionId: string;
	dimensions: number;
	maxContextMessages?: number;
	minSimilarity?: number;
	searchMode?: SearchMode;
	persistToStaticData?: boolean;
}

export interface RAGContext {
	messages: Message[];
	relevantHistory: Array<{
		content: string;
		role: string;
		similarity: number;
		metadata?: Record<string, unknown>;
	}>;
}

// In-memory storage for RAG stores
const ragStores: Map<string, {
	store: VectorStore;
	messages: Message[];
	lastAccess: number;
}> = new Map();

// TTL for RAG store entries (remove if unused for 1 hour)
const STORE_TTL = 60 * 60 * 1000; // 1 hour

/**
 * Lazy cleanup of expired stores (called on access instead of using timers)
 * n8n Cloud doesn't allow setInterval, so we clean up on each access
 */
function cleanupExpiredStores(): void {
	const now = Date.now();
	for (const [key, value] of ragStores.entries()) {
		if (now - value.lastAccess > STORE_TTL) {
			ragStores.delete(key);
		}
	}
}

export class RAGMemory {
	private config: RAGConfig;
	private executeFunctions?: IExecuteFunctions;
	private llmProvider: LLMProvider;
	private storeKey: string;

	constructor(
		config: RAGConfig,
		llmProvider: LLMProvider,
		executeFunctions?: IExecuteFunctions,
	) {
		this.config = {
			maxContextMessages: 10,
			minSimilarity: 0.5,
			searchMode: 'hybrid',
			persistToStaticData: false,
			...config,
		};
		this.llmProvider = llmProvider;
		this.executeFunctions = executeFunctions;

		const workflowId = executeFunctions?.getWorkflow().id ?? 'default';
		this.storeKey = `${workflowId}__${config.sessionId}`;
	}

	/**
	 * Gets or creates the vector store for this session
	 */
	private getStore(): { store: VectorStore; messages: Message[] } {
		// Lazy cleanup of expired stores on access
		cleanupExpiredStores();

		let entry = ragStores.get(this.storeKey);

		if (!entry) {
			// Try to load from static data if configured
			if (this.config.persistToStaticData && this.executeFunctions) {
				const loaded = this.loadFromStaticData();
				if (loaded) {
					entry = loaded;
					ragStores.set(this.storeKey, entry);
				}
			}

			if (!entry) {
				const store = new VectorStore({
					dimensions: this.config.dimensions,
					distance: 'cosine',
				});
				store.configureBM25(['content']);

				entry = {
					store,
					messages: [],
					lastAccess: Date.now(),
				};
				ragStores.set(this.storeKey, entry);
			}
		}

		entry.lastAccess = Date.now();
		return { store: entry.store, messages: entry.messages };
	}

	/**
	 * Generates an embedding for text using the LLM provider
	 */
	private async embed(text: string): Promise<number[]> {
		if (!this.llmProvider.embed) {
			throw new Error('LLM provider does not support embeddings');
		}

		const embeddings = await this.llmProvider.embed([text]);
		return embeddings[0];
	}

	/**
	 * Adds a message to the RAG memory
	 */
	async addMessage(message: Message): Promise<void> {
		const { store, messages } = this.getStore();

		// Skip system messages and empty content
		if (message.role === 'system' || !message.content?.trim()) {
			return;
		}

		// Generate embedding for the message content
		const vector = await this.embed(message.content);

		// Create document ID
		const docId = `msg_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`;

		// Add to vector store
		store.add({
			id: docId,
			content: message.content,
			vector,
			metadata: {
				role: message.role,
				timestamp: Date.now(),
				...(message.toolCallId && { toolCallId: message.toolCallId }),
			},
		});

		// Add to messages array
		messages.push(message);

		// Persist if configured
		if (this.config.persistToStaticData) {
			await this.saveToStaticData();
		}
	}

	/**
	 * Adds multiple messages to the RAG memory
	 */
	async addMessages(newMessages: Message[]): Promise<void> {
		for (const msg of newMessages) {
			await this.addMessage(msg);
		}
	}

	/**
	 * Retrieves relevant context for a query
	 */
	async getContext(query: string, k?: number): Promise<RAGContext> {
		const { store, messages } = this.getStore();
		const maxK = k ?? this.config.maxContextMessages ?? 10;

		// Get query embedding
		const queryVector = await this.embed(query);

		// Perform search based on mode
		let relevantHistory: RAGContext['relevantHistory'] = [];

		if (store.size > 0) {
			if (this.config.searchMode === 'hybrid') {
				const results = store.hybridSearch({
					mode: 'hybrid',
					k: maxK,
					queryVector,
					keywords: query,
					alpha: 0.5,
				});

				relevantHistory = results.map((r: HybridSearchResult) => ({
					content: (r.metadata?.content as string) || '',
					role: (r.metadata?.role as string) || 'unknown',
					similarity: r.vectorSimilarity ?? r.score,
					metadata: r.metadata,
				}));
			} else if (this.config.searchMode === 'keyword') {
				const results = store.keywordSearch(query, maxK);

				relevantHistory = results.map(r => {
					const doc = store.get(r.id);
					return {
						content: doc?.content || '',
						role: (doc?.metadata?.role as string) || 'unknown',
						similarity: r.score,
						metadata: r.metadata,
					};
				});
			} else {
				// Vector search
				const results = store.search(queryVector, maxK, this.config.minSimilarity);

				relevantHistory = results.map((r: SearchResult) => ({
					content: r.content,
					role: (r.metadata?.role as string) || 'unknown',
					similarity: r.similarity,
					metadata: r.metadata,
				}));
			}
		}

		// Filter by minimum similarity
		if (this.config.minSimilarity) {
			relevantHistory = relevantHistory.filter(
				(h) => h.similarity >= (this.config.minSimilarity ?? 0),
			);
		}

		return {
			messages,
			relevantHistory,
		};
	}

	/**
	 * Builds context messages to inject into the conversation
	 */
	async buildContextMessages(query: string): Promise<Message[]> {
		const context = await this.getContext(query);
		const contextMessages: Message[] = [];

		if (context.relevantHistory.length > 0) {
			// Format relevant history as a context message
			const contextText = context.relevantHistory
				.map((h, i) => `[${i + 1}] (${h.role}, similarity: ${(h.similarity * 100).toFixed(1)}%): ${h.content}`)
				.join('\n\n');

			contextMessages.push({
				role: 'system',
				content: `Relevant conversation history for context:\n\n${contextText}\n\nUse this context to inform your response if relevant.`,
			});
		}

		return contextMessages;
	}

	/**
	 * Gets all stored messages
	 */
	getMessages(): Message[] {
		const { messages } = this.getStore();
		return [...messages];
	}

	/**
	 * Clears the RAG memory
	 */
	async clear(): Promise<void> {
		ragStores.delete(this.storeKey);

		if (this.config.persistToStaticData && this.executeFunctions) {
			const staticData = this.executeFunctions.getWorkflowStaticData('global');
			delete staticData[`rag_${this.config.sessionId}`];
		}
	}

	/**
	 * Gets statistics about the RAG memory
	 */
	getStats(): {
		messageCount: number;
		documentCount: number;
		dimensions: number;
	} {
		const { store, messages } = this.getStore();
		return {
			messageCount: messages.length,
			documentCount: store.size,
			dimensions: this.config.dimensions,
		};
	}

	// Static data persistence methods

	private loadFromStaticData(): { store: VectorStore; messages: Message[]; lastAccess: number } | null {
		if (!this.executeFunctions) return null;

		const staticData = this.executeFunctions.getWorkflowStaticData('global');
		const key = `rag_${this.config.sessionId}`;
		const stored = staticData[key];

		if (typeof stored === 'string') {
			try {
				const data = JSON.parse(stored) as {
					store: {
						dimensions: number;
						distance: 'cosine' | 'euclidean' | 'dot';
						documents: VectorDocument[];
					};
					messages: Message[];
				};

				const store = VectorStore.import(data.store);
				store.configureBM25(['content']);

				return {
					store,
					messages: data.messages || [],
					lastAccess: Date.now(),
				};
			} catch {
				return null;
			}
		}

		return null;
	}

	private async saveToStaticData(): Promise<void> {
		if (!this.executeFunctions) return;

		const { store, messages } = this.getStore();
		const staticData = this.executeFunctions.getWorkflowStaticData('global');
		const key = `rag_${this.config.sessionId}`;

		const data = {
			store: store.export(),
			messages,
		};

		staticData[key] = JSON.stringify(data);
	}
}

/**
 * Gets the embedding dimensions for a provider
 */
export function getEmbeddingDimensions(provider: string, model?: string): number {
	// Default dimensions for common embedding models
	const dimensions: Record<string, number> = {
		// OpenAI
		'text-embedding-3-small': 1536,
		'text-embedding-3-large': 3072,
		'text-embedding-ada-002': 1536,
		// Gemini
		'text-embedding-004': 768,
		'embedding-001': 768,
		// Default fallbacks by provider
		'openai': 1536,
		'gemini': 768,
		'anthropic': 1024, // If using a compatible embeddings service
	};

	if (model && dimensions[model]) {
		return dimensions[model];
	}

	return dimensions[provider] || 768;
}
