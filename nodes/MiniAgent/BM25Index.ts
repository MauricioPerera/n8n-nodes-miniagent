/**
 * BM25 (Best Matching 25) keyword search implementation
 * Pure TypeScript with zero external dependencies
 */

import { tokenize, getTermFrequencies, extractTextFromMetadata, TokenizerOptions } from './TextUtils';

export interface BM25Options {
	k1?: number;           // Term saturation parameter (default: 1.2)
	b?: number;            // Length normalization parameter (default: 0.75)
	textFields: string[];  // Metadata fields to index
	tokenizerOptions?: TokenizerOptions;
}

export interface BM25SearchResult {
	id: string;
	score: number;         // BM25 score (higher = more relevant)
	metadata?: Record<string, unknown>;
}

export interface SerializedBM25Index {
	version: string;
	k1: number;
	b: number;
	textFields: string[];
	avgDocLength: number;
	documentCount: number;
	documents: Array<{
		id: string;
		length: number;
		termFrequencies: Record<string, number>;
	}>;
	documentFrequencies: Record<string, number>;
}

interface DocumentData {
	length: number;
	termFrequencies: Map<string, number>;
	metadata?: Record<string, unknown>;
}

/**
 * BM25 Index for full-text keyword search
 */
export class BM25Index {
	private k1: number;
	private b: number;
	private textFields: string[];
	private tokenizerOptions: TokenizerOptions;

	// Document storage: id -> document data
	private documents: Map<string, DocumentData> = new Map();

	// Inverted index: term -> count of documents containing this term
	private documentFrequencies: Map<string, number> = new Map();

	// Statistics
	private totalDocLength: number = 0;

	constructor(options: BM25Options) {
		this.k1 = options.k1 ?? 1.2;
		this.b = options.b ?? 0.75;
		this.textFields = options.textFields;
		this.tokenizerOptions = options.tokenizerOptions ?? {};
	}

	/**
	 * Gets the text fields being indexed
	 */
	get indexedFields(): string[] {
		return [...this.textFields];
	}

	/**
	 * Gets the number of documents in the index
	 */
	get documentCount(): number {
		return this.documents.size;
	}

	/**
	 * Gets the average document length
	 */
	get avgDocLength(): number {
		if (this.documents.size === 0) return 0;
		return this.totalDocLength / this.documents.size;
	}

	/**
	 * Gets the vocabulary size (unique terms)
	 */
	get vocabularySize(): number {
		return this.documentFrequencies.size;
	}

	/**
	 * Adds a document to the index
	 */
	addDocument(id: string, metadata: Record<string, unknown> | null): void {
		// Remove existing document if present
		if (this.documents.has(id)) {
			this.removeDocument(id);
		}

		// Extract and tokenize text
		const text = extractTextFromMetadata(metadata, this.textFields);
		const tokens = tokenize(text, this.tokenizerOptions);
		const termFrequencies = getTermFrequencies(tokens);
		const docLength = tokens.length;

		// Update document frequencies (IDF)
		const seenTerms = new Set<string>();
		for (const token of tokens) {
			if (!seenTerms.has(token)) {
				seenTerms.add(token);
				const current = this.documentFrequencies.get(token) || 0;
				this.documentFrequencies.set(token, current + 1);
			}
		}

		// Store document
		this.documents.set(id, {
			length: docLength,
			termFrequencies,
			metadata: metadata || undefined,
		});

		this.totalDocLength += docLength;
	}

	/**
	 * Updates a document in the index
	 */
	updateDocument(id: string, metadata: Record<string, unknown> | null): void {
		this.addDocument(id, metadata);
	}

	/**
	 * Removes a document from the index
	 */
	removeDocument(id: string): boolean {
		const doc = this.documents.get(id);
		if (!doc) return false;

		// Update document frequencies
		for (const [term] of doc.termFrequencies) {
			const current = this.documentFrequencies.get(term) || 0;
			if (current <= 1) {
				this.documentFrequencies.delete(term);
			} else {
				this.documentFrequencies.set(term, current - 1);
			}
		}

		// Update total length
		this.totalDocLength -= doc.length;

		// Remove document
		this.documents.delete(id);

		return true;
	}

	/**
	 * Calculates the IDF (Inverse Document Frequency) for a term
	 * Using the BM25 IDF formula: ln((N - n + 0.5) / (n + 0.5) + 1)
	 */
	private calculateIDF(term: string): number {
		const n = this.documentFrequencies.get(term) || 0;
		const N = this.documents.size;

		if (n === 0) return 0;

		// BM25 IDF formula (Robertson-Walker IDF)
		return Math.log(((N - n + 0.5) / (n + 0.5)) + 1);
	}

	/**
	 * Calculates the BM25 score for a document given query terms
	 */
	private calculateScore(docId: string, queryTerms: string[]): number {
		const doc = this.documents.get(docId);
		if (!doc) return 0;

		const avgdl = this.avgDocLength;
		if (avgdl === 0) return 0;

		let score = 0;

		for (const term of queryTerms) {
			const idf = this.calculateIDF(term);
			const tf = doc.termFrequencies.get(term) || 0;

			if (tf === 0) continue;

			// BM25 scoring formula
			const numerator = tf * (this.k1 + 1);
			const denominator = tf + this.k1 * (1 - this.b + this.b * (doc.length / avgdl));

			score += idf * (numerator / denominator);
		}

		return score;
	}

	/**
	 * Searches the index and returns k most relevant documents
	 */
	search(query: string, k: number): BM25SearchResult[] {
		if (this.documents.size === 0 || !query.trim()) {
			return [];
		}

		// Tokenize query
		const queryTerms = tokenize(query, this.tokenizerOptions);

		if (queryTerms.length === 0) {
			return [];
		}

		// Calculate scores for all documents
		const scores: Array<{ id: string; score: number }> = [];

		for (const [docId] of this.documents) {
			const score = this.calculateScore(docId, queryTerms);
			if (score > 0) {
				scores.push({ id: docId, score });
			}
		}

		// Sort by score (descending)
		scores.sort((a, b) => b.score - a.score);

		// Return top k results with metadata
		return scores.slice(0, k).map(({ id, score }) => {
			const doc = this.documents.get(id);
			return {
				id,
				score,
				metadata: doc?.metadata,
			};
		});
	}

	/**
	 * Gets statistics about the index
	 */
	getStats(): {
		documentCount: number;
		avgDocLength: number;
		vocabularySize: number;
		k1: number;
		b: number;
		textFields: string[];
	} {
		return {
			documentCount: this.documentCount,
			avgDocLength: this.avgDocLength,
			vocabularySize: this.vocabularySize,
			k1: this.k1,
			b: this.b,
			textFields: this.textFields,
		};
	}

	/**
	 * Serializes the index to a plain object
	 */
	serialize(): SerializedBM25Index {
		const documents: SerializedBM25Index['documents'] = [];

		for (const [id, doc] of this.documents) {
			const termFrequencies: Record<string, number> = {};
			for (const [term, freq] of doc.termFrequencies) {
				termFrequencies[term] = freq;
			}
			documents.push({
				id,
				length: doc.length,
				termFrequencies,
			});
		}

		const documentFrequencies: Record<string, number> = {};
		for (const [term, freq] of this.documentFrequencies) {
			documentFrequencies[term] = freq;
		}

		return {
			version: '1.0.0',
			k1: this.k1,
			b: this.b,
			textFields: this.textFields,
			avgDocLength: this.avgDocLength,
			documentCount: this.documentCount,
			documents,
			documentFrequencies,
		};
	}

	/**
	 * Deserializes an index from a plain object
	 */
	static deserialize(data: SerializedBM25Index): BM25Index {
		const index = new BM25Index({
			k1: data.k1,
			b: data.b,
			textFields: data.textFields,
		});

		// Restore documents
		for (const doc of data.documents) {
			const termFrequencies = new Map<string, number>();
			for (const [term, freq] of Object.entries(doc.termFrequencies)) {
				termFrequencies.set(term, freq);
			}

			index.documents.set(doc.id, {
				length: doc.length,
				termFrequencies,
			});

			index.totalDocLength += doc.length;
		}

		// Restore document frequencies
		for (const [term, freq] of Object.entries(data.documentFrequencies)) {
			index.documentFrequencies.set(term, freq);
		}

		return index;
	}

	/**
	 * Clears the entire index
	 */
	clear(): void {
		this.documents.clear();
		this.documentFrequencies.clear();
		this.totalDocLength = 0;
	}
}
