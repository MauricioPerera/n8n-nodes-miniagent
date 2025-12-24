/**
 * Text processing utilities for BM25 keyword search
 */

export interface TokenizerOptions {
	lowercase?: boolean;
	removePunctuation?: boolean;
	removeNumbers?: boolean;
	minTokenLength?: number;
}

const DEFAULT_OPTIONS: TokenizerOptions = {
	lowercase: true,
	removePunctuation: true,
	removeNumbers: false,
	minTokenLength: 1,
};

/**
 * Tokenizes text into normalized tokens
 */
export function tokenize(text: string, options?: TokenizerOptions): string[] {
	const opts = { ...DEFAULT_OPTIONS, ...options };

	if (!text || typeof text !== 'string') {
		return [];
	}

	let processed = text;

	// Convert to lowercase
	if (opts.lowercase) {
		processed = processed.toLowerCase();
	}

	// Remove punctuation (keep alphanumeric and spaces)
	if (opts.removePunctuation) {
		processed = processed.replace(/[^\p{L}\p{N}\s]/gu, ' ');
	}

	// Split on whitespace
	let tokens = processed.split(/\s+/).filter(t => t.length > 0);

	// Remove numbers if requested
	if (opts.removeNumbers) {
		tokens = tokens.filter(t => !/^\d+$/.test(t));
	}

	// Filter by minimum length
	if (opts.minTokenLength && opts.minTokenLength > 1) {
		const minLen = opts.minTokenLength;
		tokens = tokens.filter(t => t.length >= minLen);
	}

	return tokens;
}

/**
 * Creates term frequency map for a document
 */
export function getTermFrequencies(tokens: string[]): Map<string, number> {
	const frequencies = new Map<string, number>();

	for (const token of tokens) {
		const current = frequencies.get(token) || 0;
		frequencies.set(token, current + 1);
	}

	return frequencies;
}

/**
 * Gets a nested value from an object using dot notation
 */
function getNestedValue(obj: Record<string, unknown>, path: string): unknown {
	const parts = path.split('.');
	let current: unknown = obj;

	for (const part of parts) {
		if (current === null || current === undefined) return undefined;
		if (typeof current !== 'object') return undefined;
		current = (current as Record<string, unknown>)[part];
	}

	return current;
}

/**
 * Extracts text from metadata fields and concatenates them
 */
export function extractTextFromMetadata(
	metadata: Record<string, unknown> | null,
	textFields: string[],
): string {
	if (!metadata) {
		return '';
	}

	const textParts: string[] = [];

	for (const field of textFields) {
		const value = getNestedValue(metadata, field);

		if (typeof value === 'string') {
			textParts.push(value);
		} else if (Array.isArray(value)) {
			// Handle arrays of strings (e.g., tags)
			const stringValues = value.filter(v => typeof v === 'string');
			textParts.push(...stringValues);
		}
	}

	return textParts.join(' ');
}
