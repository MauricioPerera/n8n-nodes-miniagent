import type {
	ICredentialType,
	INodeProperties,
} from 'n8n-workflow';

export class GeminiApi implements ICredentialType {
	name = 'geminiApi';
	displayName = 'Gemini API';
	documentationUrl = 'https://ai.google.dev/docs';
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
			description: 'Your Google AI Studio API key. Get it from https://aistudio.google.com/app/apikey',
		},
	];
}
