/**
 * Tool Executor
 * Executes tools defined as JSON schemas
 * Supports two types of tools:
 * 1. Code tools - Execute JavaScript code in a sandbox
 * 2. HTTP tools - Make HTTP requests to external APIs
 */

import type { ToolDefinition, ToolCall } from './LLMProvider';

// Extended tool definition with execution config
export interface ExecutableTool extends ToolDefinition {
	// For code execution
	code?: string;
	// For HTTP requests
	http?: {
		url: string;
		method: 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE';
		headers?: Record<string, string>;
		body?: string;
		queryParams?: Record<string, string>;
	};
}

export interface ToolResult {
	success: boolean;
	output: string;
	error?: string;
}

export class ToolExecutor {
	private tools: Map<string, ExecutableTool>;

	constructor(tools: ExecutableTool[]) {
		this.tools = new Map();
		for (const tool of tools) {
			this.tools.set(tool.name, tool);
		}
	}

	/**
	 * Get tool definitions for the LLM (without execution config)
	 */
	getDefinitions(): ToolDefinition[] {
		return Array.from(this.tools.values()).map((tool) => ({
			name: tool.name,
			description: tool.description,
			parameters: tool.parameters,
		}));
	}

	/**
	 * Check if a tool exists
	 */
	hasTool(name: string): boolean {
		return this.tools.has(name);
	}

	/**
	 * Execute a tool call
	 */
	async execute(toolCall: ToolCall): Promise<ToolResult> {
		const tool = this.tools.get(toolCall.name);

		if (!tool) {
			return {
				success: false,
				output: '',
				error: `Tool "${toolCall.name}" not found`,
			};
		}

		try {
			if (tool.http) {
				return await this.executeHttp(tool.http, toolCall.arguments);
			} else if (tool.code) {
				return await this.executeCode(tool.code, toolCall.arguments);
			} else {
				return {
					success: false,
					output: '',
					error: `Tool "${toolCall.name}" has no execution config (code or http)`,
				};
			}
		} catch (error) {
			return {
				success: false,
				output: '',
				error: error instanceof Error ? error.message : String(error),
			};
		}
	}

	/**
	 * Execute an HTTP tool
	 */
	private async executeHttp(
		config: ExecutableTool['http'],
		args: Record<string, unknown>,
	): Promise<ToolResult> {
		if (!config) {
			throw new Error('HTTP config is undefined');
		}

		// Replace placeholders in URL and body
		let url = this.replacePlaceholders(config.url, args);
		let body = config.body ? this.replacePlaceholders(config.body, args) : undefined;

		// Build query params
		if (config.queryParams) {
			const params = new URLSearchParams();
			for (const [key, value] of Object.entries(config.queryParams)) {
				const replaced = this.replacePlaceholders(value, args);
				params.append(key, replaced);
			}
			const queryString = params.toString();
			if (queryString) {
				url += (url.includes('?') ? '&' : '?') + queryString;
			}
		}

		// Build headers
		const headers: Record<string, string> = {
			'Content-Type': 'application/json',
			...config.headers,
		};

		// Replace placeholders in headers
		for (const [key, value] of Object.entries(headers)) {
			headers[key] = this.replacePlaceholders(value, args);
		}

		// Make the request
		const response = await fetch(url, {
			method: config.method,
			headers,
			body: config.method !== 'GET' ? body : undefined,
		});

		const responseText = await response.text();

		if (!response.ok) {
			return {
				success: false,
				output: responseText,
				error: `HTTP ${response.status}: ${response.statusText}`,
			};
		}

		// Try to parse as JSON for cleaner output
		try {
			const json = JSON.parse(responseText);
			return {
				success: true,
				output: JSON.stringify(json, null, 2),
			};
		} catch {
			return {
				success: true,
				output: responseText,
			};
		}
	}

	/**
	 * Execute a code tool in a sandboxed environment
	 */
	private async executeCode(
		code: string,
		args: Record<string, unknown>,
	): Promise<ToolResult> {
		// Create a sandboxed function
		// The code has access to the args and can use basic JavaScript
		// No access to require, process, global, etc.

		const sandboxedCode = `
			"use strict";
			const args = ${JSON.stringify(args)};
			${this.extractArgVariables(args)}
			${code}
		`;

		try {
			// Use Function constructor as a basic sandbox
			// Note: This is not a perfect sandbox, but provides basic isolation
			const fn = new Function(sandboxedCode);
			const result = fn();

			// Handle async results
			const output = result instanceof Promise ? await result : result;

			return {
				success: true,
				output: typeof output === 'string' ? output : JSON.stringify(output),
			};
		} catch (error) {
			return {
				success: false,
				output: '',
				error: error instanceof Error ? error.message : String(error),
			};
		}
	}

	/**
	 * Replace {{placeholder}} with values from args
	 */
	private replacePlaceholders(template: string, args: Record<string, unknown>): string {
		return template.replace(/\{\{(\w+)\}\}/g, (_, key) => {
			const value = args[key];
			if (value === undefined) return '';
			return typeof value === 'string' ? value : JSON.stringify(value);
		});
	}

	/**
	 * Extract args as individual variables for code execution
	 */
	private extractArgVariables(args: Record<string, unknown>): string {
		return Object.entries(args)
			.map(([key, value]) => `const ${key} = ${JSON.stringify(value)};`)
			.join('\n');
	}
}

/**
 * Parse tools from JSON string or array
 */
export function parseTools(input: string | ExecutableTool[]): ExecutableTool[] {
	if (Array.isArray(input)) {
		return input;
	}

	try {
		const parsed = JSON.parse(input);
		return Array.isArray(parsed) ? parsed : [];
	} catch {
		return [];
	}
}

/**
 * Validate a tool definition
 */
export function validateTool(tool: unknown): tool is ExecutableTool {
	if (!tool || typeof tool !== 'object') return false;

	const t = tool as Record<string, unknown>;

	if (typeof t.name !== 'string' || !t.name) return false;
	if (typeof t.description !== 'string') return false;
	if (!t.parameters || typeof t.parameters !== 'object') return false;

	// Must have either code or http
	const hasCode = typeof t.code === 'string';
	const hasHttp = t.http !== undefined && typeof t.http === 'object';

	return hasCode || hasHttp ? true : false;
}
