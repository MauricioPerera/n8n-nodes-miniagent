import type {
	ICredentialType,
	INodeProperties,
} from 'n8n-workflow';

export class OpenAiCompatibleApi implements ICredentialType {
	name = 'openAiCompatibleApi';
	displayName = 'OpenAI Compatible API';
	documentationUrl = 'https://platform.openai.com/docs/api-reference';
	properties: INodeProperties[] = [
		{
			displayName: 'API Key',
			name: 'apiKey',
			type: 'string',
			typeOptions: {
				password: true,
			},
			default: '',
			required: true,
			description: 'Your API key for the OpenAI-compatible service',
		},
		{
			displayName: 'Base URL',
			name: 'baseUrl',
			type: 'string',
			default: 'https://api.openai.com/v1',
			required: true,
			description: 'The base URL of the API. Examples: https://api.openai.com/v1 (OpenAI), https://openrouter.ai/api/v1 (OpenRouter), https://api.groq.com/openai/v1 (Groq), http://localhost:11434/v1 (Ollama)',
		},
	];
}
