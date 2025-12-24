/**
 * VectorStore - Simplified vector storage for RAG memory
 * Pure TypeScript implementation with hybrid search support
 */

import { BM25Index, BM25SearchResult } from './BM25Index';
import { SearchMode, FusionMethod, HybridSearchResult, VectorSearchResult, hybridFusion } from './HybridSearch';

export type DistanceMetric = 'cosine' | 'euclidean' | 'dot';

export interface VectorStoreOptions {
	dimensions: number;
	distance?: DistanceMetric;
}

export interface VectorDocument {
	id: string;
	content: string;
	vector: number[];
	metadata?: Record<string, unknown>;
}

export interface SearchResult {
	id: string;
	content: string;
	distance: number;
	similarity: number;
	metadata?: Record<string, unknown>;
}

export interface HybridSearchOptions {
	mode: SearchMode;
	k: number;
	queryVector?: number[];
	keywords?: string;
	alpha?: number;
	fusionMethod?: FusionMethod;
}

interface StoredDocument {
	id: string;
	content: string;
	vector: number[];
	metadata: Record<string, unknown> | null;
	norm?: number;
}

/**
 * Calculates cosine distance between two vectors
 */
function cosineDistance(a: number[], b: number[], normA?: number, normB?: number): number {
	let dot = 0;
	let nA = normA ?? 0;
	let nB = normB ?? 0;

	const needNormA = normA === undefined;
	const needNormB = normB === undefined;

	for (let i = 0; i < a.length; i++) {
		dot += a[i] * b[i];
		if (needNormA) nA += a[i] * a[i];
		if (needNormB) nB += b[i] * b[i];
	}

	if (needNormA) nA = Math.sqrt(nA);
	if (needNormB) nB = Math.sqrt(nB);

	const denom = nA * nB;
	if (denom === 0) return 1;

	const similarity = Math.max(-1, Math.min(1, dot / denom));
	return 1 - similarity;
}

/**
 * Calculates euclidean distance between two vectors
 */
function euclideanDistance(a: number[], b: number[]): number {
	let sum = 0;
	for (let i = 0; i < a.length; i++) {
		const diff = a[i] - b[i];
		sum += diff * diff;
	}
	return Math.sqrt(sum);
}

/**
 * Calculates dot product distance
 */
function dotProductDistance(a: number[], b: number[]): number {
	let dot = 0;
	for (let i = 0; i < a.length; i++) {
		dot += a[i] * b[i];
	}
	return -dot;
}

/**
 * In-memory vector store for RAG
 */
export class VectorStore {
	private documents: Map<string, StoredDocument> = new Map();
	private readonly dimensions: number;
	private readonly distance: DistanceMetric;
	private bm25Index: BM25Index | null = null;

	constructor(options: VectorStoreOptions) {
		this.dimensions = options.dimensions;
		this.distance = options.distance || 'cosine';
	}

	/**
	 * Gets the number of documents
	 */
	get size(): number {
		return this.documents.size;
	}

	/**
	 * Computes the norm of a vector
	 */
	private computeNorm(vector: number[]): number {
		let sum = 0;
		for (let i = 0; i < vector.length; i++) {
			sum += vector[i] * vector[i];
		}
		return Math.sqrt(sum);
	}

	/**
	 * Calculates distance using the configured metric
	 */
	private calculateDistance(a: number[], b: number[], normA?: number, normB?: number): number {
		switch (this.distance) {
			case 'cosine':
				return cosineDistance(a, b, normA, normB);
			case 'euclidean':
				return euclideanDistance(a, b);
			case 'dot':
				return dotProductDistance(a, b);
			default:
				return cosineDistance(a, b, normA, normB);
		}
	}

	/**
	 * Adds a document with its embedding
	 */
	add(doc: VectorDocument): void {
		if (doc.vector.length !== this.dimensions) {
			throw new Error(`Dimension mismatch: expected ${this.dimensions}, got ${doc.vector.length}`);
		}

		const norm = this.distance === 'cosine' ? this.computeNorm(doc.vector) : undefined;

		this.documents.set(doc.id, {
			id: doc.id,
			content: doc.content,
			vector: [...doc.vector],
			metadata: doc.metadata || null,
			norm,
		});

		// Update BM25 index
		if (this.bm25Index) {
			this.bm25Index.addDocument(doc.id, { content: doc.content, ...doc.metadata });
		}
	}

	/**
	 * Adds multiple documents
	 */
	addMany(docs: VectorDocument[]): void {
		for (const doc of docs) {
			this.add(doc);
		}
	}

	/**
	 * Removes a document by ID
	 */
	remove(id: string): boolean {
		const deleted = this.documents.delete(id);
		if (deleted && this.bm25Index) {
			this.bm25Index.removeDocument(id);
		}
		return deleted;
	}

	/**
	 * Clears all documents
	 */
	clear(): void {
		this.documents.clear();
		if (this.bm25Index) {
			this.bm25Index.clear();
		}
	}

	/**
	 * Configures BM25 for keyword search
	 */
	configureBM25(textFields: string[] = ['content']): void {
		this.bm25Index = new BM25Index({ textFields });

		// Index existing documents
		for (const doc of this.documents.values()) {
			this.bm25Index.addDocument(doc.id, { content: doc.content, ...doc.metadata });
		}
	}

	/**
	 * Performs vector similarity search
	 */
	search(queryVector: number[], k: number, minSimilarity: number = 0): SearchResult[] {
		if (queryVector.length !== this.dimensions) {
			throw new Error(`Query dimension mismatch: expected ${this.dimensions}, got ${queryVector.length}`);
		}

		if (this.documents.size === 0) {
			return [];
		}

		const queryNorm = this.distance === 'cosine' ? this.computeNorm(queryVector) : undefined;
		const results: SearchResult[] = [];

		for (const doc of this.documents.values()) {
			const distance = this.calculateDistance(queryVector, doc.vector, queryNorm, doc.norm);

			let similarity: number;
			if (this.distance === 'cosine') {
				similarity = 1 - distance;
			} else if (this.distance === 'dot') {
				similarity = -distance;
			} else {
				similarity = 1 / (1 + distance);
			}

			if (similarity < minSimilarity) {
				continue;
			}

			results.push({
				id: doc.id,
				content: doc.content,
				distance,
				similarity,
				metadata: doc.metadata || undefined,
			});
		}

		results.sort((a, b) => a.distance - b.distance);
		return results.slice(0, Math.min(k, results.length));
	}

	/**
	 * Performs keyword search using BM25
	 */
	keywordSearch(query: string, k: number): BM25SearchResult[] {
		if (!this.bm25Index) {
			this.configureBM25();
		}
		return this.bm25Index!.search(query, k);
	}

	/**
	 * Performs hybrid search combining vector and keyword search
	 */
	hybridSearch(options: HybridSearchOptions): HybridSearchResult[] {
		const { mode, k, queryVector, keywords, alpha = 0.5, fusionMethod = 'rrf' } = options;

		if (mode === 'vector') {
			if (!queryVector) {
				throw new Error('queryVector is required for vector search mode');
			}
			return this.search(queryVector, k).map(r => ({
				id: r.id,
				score: r.similarity,
				vectorSimilarity: r.similarity,
				metadata: { ...r.metadata, content: r.content },
			}));
		}

		if (mode === 'keyword') {
			if (!keywords) {
				throw new Error('keywords is required for keyword search mode');
			}
			const results = this.keywordSearch(keywords, k);
			return results.map(r => {
				const doc = this.documents.get(r.id);
				return {
					id: r.id,
					score: r.score,
					keywordScore: r.score,
					metadata: { ...r.metadata, content: doc?.content },
				};
			});
		}

		// Hybrid mode
		if (!queryVector) {
			throw new Error('queryVector is required for hybrid search mode');
		}
		if (!keywords) {
			throw new Error('keywords is required for hybrid search mode');
		}

		const fetchK = Math.max(k * 3, 50);
		const vectorResults = this.search(queryVector, fetchK);
		const keywordResults = this.keywordSearch(keywords, fetchK);

		const vectorForFusion: VectorSearchResult[] = vectorResults.map(r => ({
			id: r.id,
			distance: r.distance,
			similarity: r.similarity,
			metadata: { ...r.metadata, content: r.content },
		}));

		return hybridFusion(vectorForFusion, keywordResults, k, fusionMethod, { alpha });
	}

	/**
	 * Gets a document by ID
	 */
	get(id: string): VectorDocument | null {
		const doc = this.documents.get(id);
		if (!doc) return null;

		return {
			id: doc.id,
			content: doc.content,
			vector: [...doc.vector],
			metadata: doc.metadata || undefined,
		};
	}

	/**
	 * Exports the store to a serializable object
	 */
	export(): {
		dimensions: number;
		distance: DistanceMetric;
		documents: VectorDocument[];
	} {
		return {
			dimensions: this.dimensions,
			distance: this.distance,
			documents: Array.from(this.documents.values()).map(doc => ({
				id: doc.id,
				content: doc.content,
				vector: doc.vector,
				metadata: doc.metadata || undefined,
			})),
		};
	}

	/**
	 * Imports documents from a serialized object
	 */
	static import(data: {
		dimensions: number;
		distance: DistanceMetric;
		documents: VectorDocument[];
	}): VectorStore {
		const store = new VectorStore({
			dimensions: data.dimensions,
			distance: data.distance,
		});

		for (const doc of data.documents) {
			store.add(doc);
		}

		return store;
	}
}
