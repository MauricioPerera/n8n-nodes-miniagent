/**
 * Hybrid Search - Combines vector similarity and keyword (BM25) search
 */

import { BM25SearchResult } from './BM25Index';

export type SearchMode = 'vector' | 'keyword' | 'hybrid';
export type FusionMethod = 'rrf' | 'weighted';

export interface VectorSearchResult {
	id: string;
	distance: number;
	similarity: number;
	metadata?: Record<string, unknown>;
}

export interface HybridSearchResult {
	id: string;
	score: number;              // Combined/fusion score
	vectorRank?: number;        // Rank in vector results (1-based)
	keywordRank?: number;       // Rank in keyword results (1-based)
	vectorSimilarity?: number;  // Original vector similarity
	keywordScore?: number;      // Original BM25 score
	metadata?: Record<string, unknown>;
}

/**
 * Performs Reciprocal Rank Fusion (RRF) on two result sets
 *
 * RRF is preferred for hybrid search because:
 * 1. It does not require score normalization
 * 2. It is robust to score distribution differences
 * 3. It handles missing documents gracefully
 *
 * Formula: RRF_score(d) = 1/(k + rank_vector(d)) + 1/(k + rank_keyword(d))
 *
 * @param vectorResults - Results from vector similarity search
 * @param keywordResults - Results from BM25 keyword search
 * @param k - Number of results to return
 * @param rrfConstant - RRF constant (default: 60, higher = less emphasis on top ranks)
 */
export function reciprocalRankFusion(
	vectorResults: VectorSearchResult[],
	keywordResults: BM25SearchResult[],
	k: number,
	rrfConstant: number = 60,
): HybridSearchResult[] {
	// Build rank maps
	const vectorRanks = new Map<string, number>();
	const keywordRanks = new Map<string, number>();

	vectorResults.forEach((result, index) => {
		vectorRanks.set(result.id, index + 1); // 1-based ranking
	});

	keywordResults.forEach((result, index) => {
		keywordRanks.set(result.id, index + 1); // 1-based ranking
	});

	// Collect all unique document IDs
	const allIds = new Set<string>([
		...vectorRanks.keys(),
		...keywordRanks.keys(),
	]);

	// Calculate RRF scores
	const scores: HybridSearchResult[] = [];

	for (const id of allIds) {
		const vectorRank = vectorRanks.get(id);
		const keywordRank = keywordRanks.get(id);

		// RRF score contribution from each result set
		let score = 0;
		if (vectorRank !== undefined) {
			score += 1 / (rrfConstant + vectorRank);
		}
		if (keywordRank !== undefined) {
			score += 1 / (rrfConstant + keywordRank);
		}

		// Find original result data
		const vectorResult = vectorResults.find(r => r.id === id);
		const keywordResult = keywordResults.find(r => r.id === id);

		scores.push({
			id,
			score,
			vectorRank,
			keywordRank,
			vectorSimilarity: vectorResult?.similarity,
			keywordScore: keywordResult?.score,
			metadata: vectorResult?.metadata || keywordResult?.metadata,
		});
	}

	// Sort by RRF score (descending)
	scores.sort((a, b) => b.score - a.score);

	// Return top k
	return scores.slice(0, k);
}

/**
 * Normalizes scores to [0, 1] range using min-max normalization
 */
function normalizeScores(
	results: Array<{ id: string; score: number }>,
): Map<string, number> {
	if (results.length === 0) {
		return new Map();
	}

	const scores = results.map(r => r.score);
	const minScore = Math.min(...scores);
	const maxScore = Math.max(...scores);
	const range = maxScore - minScore;

	const normalized = new Map<string, number>();

	for (const result of results) {
		const normScore = range === 0 ? 1 : (result.score - minScore) / range;
		normalized.set(result.id, normScore);
	}

	return normalized;
}

/**
 * Performs weighted score combination
 *
 * This method normalizes scores from both search types to [0, 1] range
 * and combines them using the alpha parameter.
 *
 * Formula: combined_score = alpha * vector_score + (1 - alpha) * keyword_score
 *
 * @param vectorResults - Results from vector similarity search
 * @param keywordResults - Results from BM25 keyword search
 * @param k - Number of results to return
 * @param alpha - Balance weight: 0 = pure keyword, 1 = pure vector (default: 0.5)
 */
export function weightedCombination(
	vectorResults: VectorSearchResult[],
	keywordResults: BM25SearchResult[],
	k: number,
	alpha: number = 0.5,
): HybridSearchResult[] {
	// Clamp alpha to [0, 1]
	alpha = Math.max(0, Math.min(1, alpha));

	// Normalize vector scores (similarity is already [0, 1] for cosine)
	const normalizedVector = new Map<string, number>();
	for (const result of vectorResults) {
		// Use similarity directly for vector results
		normalizedVector.set(result.id, result.similarity);
	}

	// Normalize BM25 scores (need min-max normalization)
	const normalizedKeyword = normalizeScores(
		keywordResults.map(r => ({ id: r.id, score: r.score })),
	);

	// Collect all unique document IDs
	const allIds = new Set<string>([
		...normalizedVector.keys(),
		...normalizedKeyword.keys(),
	]);

	// Calculate combined scores
	const scores: HybridSearchResult[] = [];

	for (const id of allIds) {
		const vectorScore = normalizedVector.get(id) ?? 0;
		const keywordScore = normalizedKeyword.get(id) ?? 0;

		// Weighted combination
		const combinedScore = alpha * vectorScore + (1 - alpha) * keywordScore;

		// Find original result data for metadata and original scores
		const vectorResult = vectorResults.find(r => r.id === id);
		const keywordResult = keywordResults.find(r => r.id === id);

		// Calculate ranks
		const vectorRank = vectorResults.findIndex(r => r.id === id);
		const keywordRank = keywordResults.findIndex(r => r.id === id);

		scores.push({
			id,
			score: combinedScore,
			vectorRank: vectorRank >= 0 ? vectorRank + 1 : undefined,
			keywordRank: keywordRank >= 0 ? keywordRank + 1 : undefined,
			vectorSimilarity: vectorResult?.similarity,
			keywordScore: keywordResult?.score,
			metadata: vectorResult?.metadata || keywordResult?.metadata,
		});
	}

	// Sort by combined score (descending)
	scores.sort((a, b) => b.score - a.score);

	// Return top k
	return scores.slice(0, k);
}

/**
 * Performs hybrid search using the specified fusion method
 */
export function hybridFusion(
	vectorResults: VectorSearchResult[],
	keywordResults: BM25SearchResult[],
	k: number,
	method: FusionMethod = 'rrf',
	options?: {
		alpha?: number;       // For weighted method
		rrfConstant?: number; // For RRF method
	},
): HybridSearchResult[] {
	if (method === 'rrf') {
		return reciprocalRankFusion(
			vectorResults,
			keywordResults,
			k,
			options?.rrfConstant ?? 60,
		);
	} else {
		return weightedCombination(
			vectorResults,
			keywordResults,
			k,
			options?.alpha ?? 0.5,
		);
	}
}
