import {
	BuiltInTool,
	MessageRole,
	ToolCallType,
	ToolPermissionDecision,
	ToolSource
} from '$lib/enums';
import { ChatService } from '$lib/services';
import { ToolsService } from '$lib/services/tools.service';
import { agenticStore } from '$lib/stores/agentic/index.svelte';
import { mcpStore } from '$lib/stores/mcp/index.svelte';
import { permissionsStore } from '$lib/stores/permissions.svelte';
import { settingsStore } from '$lib/stores/settings/index.svelte';
import { toolsStore } from '$lib/stores/tools.svelte';
import type {
	AgenticFlowCallbacks,
	ApiChatCompletionToolCall,
	DatabaseMessage,
	ToolExecutionResult
} from '$lib/types';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('$lib/services', () => ({
	ChatService: {
		normalizeMessagesForApi: vi.fn(async (messages) => messages),
		sendMessage: vi.fn()
	}
}));
vi.mock('$lib/services/tools.service', () => ({ ToolsService: { executeTool: vi.fn() } }));
vi.mock('$lib/services/read-media.service', () => ({ ReadMediaService: {} }));
vi.mock('$lib/services/sandbox.service', () => ({ SandboxService: {} }));
vi.mock('$lib/utils', async () => ({
	...(await import('$lib/utils/abort')),
	executeBrowserInfoTool: vi.fn(),
	executeGetDatetimeTool: vi.fn(),
	getAudioInputFormat: vi.fn()
}));
vi.mock('$lib/stores/conversations/index.svelte', () => ({
	conversationsStore: {
		onConversationsDeleted: vi.fn(),
		preferences: { policyEnabledServerIds: () => ['server-a', 'server-b'] }
	}
}));
vi.mock('$lib/stores/mcp/index.svelte', () => ({
	mcpStore: {
		acquireConnection: vi.fn(),
		ensureInitialized: vi.fn(async () => true),
		executeTool: vi.fn(),
		getToolServer: vi.fn(),
		hasEnabledServers: () => true,
		releaseConnection: vi.fn(async () => {})
	}
}));
vi.mock('$lib/stores/models/index.svelte', () => ({ modelsStore: { models: [] } }));
vi.mock('$lib/stores/permissions.svelte', () => ({
	permissionsStore: { allowTool: vi.fn(), allowTools: vi.fn(), hasTool: vi.fn() }
}));
vi.mock('$lib/stores/settings/index.svelte', () => ({
	settingsStore: { config: { agenticMaxTurns: 10, agenticParallelToolCalls: false } }
}));
vi.mock('$lib/stores/tools.svelte', () => ({
	toolsStore: {
		allTools: [],
		browserTools: [],
		customTools: [],
		disabledToolCategories: [],
		disabledTools: [],
		getEnabledToolsForLLM: () => [{ function: { name: 'ask_a' }, type: 'function' }],
		getPermissionKey: (name: string) => name,
		getToolServerLabel: (name: string) => name,
		getToolSource: vi.fn(),
		isToolsEndpointUnreachable: true,
		serverTools: []
	}
}));

const conversationId = 'tool-execution';
const executeMcp = vi.mocked(mcpStore.executeTool);
const sendMessage = vi.mocked(ChatService.sendMessage);

function deferred<T>() {
	let resolve!: (value: T) => void;
	let reject!: (reason: unknown) => void;

	const promise = new Promise<T>((resolvePromise, rejectPromise) => {
		resolve = resolvePromise;
		reject = rejectPromise;
	});

	return { promise, reject, resolve };
}

function result(content: string, isError = false): ToolExecutionResult {
	return { content, isError };
}

function start(names: string[], callbacks: AgenticFlowCallbacks = {}, signal?: AbortSignal) {
	const calls: ApiChatCompletionToolCall[] = names.map((name, i) => ({
		function: { arguments: '{}', name },
		id: `call-${i}`,
		type: ToolCallType.FUNCTION
	}));
	const createToolResultMessage = vi.fn(
		async (id: string, content: string) =>
			({
				content,
				id: `message-${id}`
			}) as DatabaseMessage
	);
	const onFlowComplete = vi.fn<NonNullable<AgenticFlowCallbacks['onFlowComplete']>>();

	sendMessage.mockImplementationOnce(async (_messages, options) => {
		options.onToolCallChunk?.(JSON.stringify(calls));
	});
	sendMessage.mockImplementationOnce(async (_messages, options) => {
		options.onChunk?.('Done');
	});

	const flow = agenticStore.runAgenticFlow({
		callbacks: { createToolResultMessage, onFlowComplete, ...callbacks },
		conversationId,
		messages: [{ content: 'Compare the answers', role: MessageRole.USER }],
		signal
	});

	return { createToolResultMessage, flow, onFlowComplete };
}

beforeEach(() => {
	vi.clearAllMocks();
	vi.useFakeTimers();
	agenticStore.clearSession(conversationId);
	settingsStore.config.agenticParallelToolCalls = false;
	vi.mocked(permissionsStore.hasTool).mockReturnValue(true);
	vi.mocked(toolsStore.getToolSource).mockImplementation((name) =>
		name === BuiltInTool.SERVER_READ_FILE ? ToolSource.SERVER : ToolSource.MCP
	);
	vi.mocked(mcpStore.getToolServer).mockImplementation((name) => name.split('_')[1]);
	executeMcp.mockReset();
	sendMessage.mockReset();
	vi.spyOn(console, 'log').mockImplementation(() => {});
});

afterEach(() => {
	vi.useRealTimers();
	vi.restoreAllMocks();
});

describe('agentic tool execution', () => {
	it('starts different MCP servers together and persists results serially in model order', async () => {
		settingsStore.config.agenticParallelToolCalls = true;
		let now = 0;

		vi.spyOn(performance, 'now').mockImplementation(() => now);
		const first = deferred<ToolExecutionResult>();
		const second = deferred<ToolExecutionResult>();
		const persisted = deferred<DatabaseMessage>();
		const save = vi
			.fn<NonNullable<AgenticFlowCallbacks['createToolResultMessage']>>()
			.mockImplementationOnce(() => persisted.promise)
			.mockResolvedValue({ id: 'second-result' } as DatabaseMessage);

		executeMcp
			.mockImplementationOnce(() => first.promise)
			.mockImplementationOnce(() => second.promise);
		const { flow, onFlowComplete } = start(['ask_a', 'ask_b'], { createToolResultMessage: save });

		await vi.runAllTimersAsync();

		expect(executeMcp).toHaveBeenCalledTimes(2);
		expect(agenticStore.getExecutingToolCallIds(conversationId)).toEqual(['call-0', 'call-1']);
		now = 20;
		second.resolve(result('second'));
		await vi.runAllTimersAsync();
		expect(save).not.toHaveBeenCalled();
		expect(sendMessage).toHaveBeenCalledTimes(1);
		expect(agenticStore.getExecutingToolCallIds(conversationId)).toEqual(['call-0']);

		now = 50;
		first.resolve(result('first'));
		await vi.runAllTimersAsync();
		expect(save).toHaveBeenCalledTimes(1);
		expect(save).toHaveBeenNthCalledWith(1, 'call-0', 'first', undefined);
		expect(sendMessage).toHaveBeenCalledTimes(1);
		now = 150;
		persisted.resolve({ id: 'first-result' } as DatabaseMessage);
		await flow;

		expect(save).toHaveBeenNthCalledWith(2, 'call-1', 'second', undefined);
		expect(
			sendMessage.mock.calls[1][0].filter((message) => message.role === MessageRole.TOOL)
		).toEqual([
			{ content: 'first', role: MessageRole.TOOL, tool_call_id: 'call-0' },
			{ content: 'second', role: MessageRole.TOOL, tool_call_id: 'call-1' }
		]);
		expect(agenticStore.getExecutingToolCallIds(conversationId)).toEqual([]);
		expect(onFlowComplete.mock.calls[0][0]?.agentic?.toolsMs).toBe(50);
		expect(
			onFlowComplete.mock.calls[0][0]?.agentic?.toolCalls?.map((call) => call.duration_ms)
		).toEqual([50, 20]);
	});

	it('keeps different MCP servers serial when parallel execution is disabled', async () => {
		const first = deferred<ToolExecutionResult>();

		executeMcp.mockImplementationOnce(() => first.promise).mockResolvedValue(result('second'));
		const { createToolResultMessage, flow } = start(['ask_a', 'ask_b']);

		await vi.runAllTimersAsync();

		expect(executeMcp).toHaveBeenCalledTimes(1);
		first.resolve(result('first'));
		await vi.runAllTimersAsync();
		await flow;

		expect(executeMcp).toHaveBeenCalledTimes(2);
		expect(createToolResultMessage.mock.invocationCallOrder[0]).toBeLessThan(
			executeMcp.mock.invocationCallOrder[1]
		);
	});

	it('resolves each approval before starting the batch and never executes denied calls', async () => {
		settingsStore.config.agenticParallelToolCalls = true;
		vi.mocked(permissionsStore.hasTool).mockReturnValue(false);
		executeMcp.mockResolvedValue(result('allowed'));
		const { createToolResultMessage, flow, onFlowComplete } = start(['ask_a', 'ask_b']);

		await vi.runAllTimersAsync();

		expect(agenticStore.getPendingPermissionRequest(conversationId)?.toolName).toBe('ask_a');
		expect(executeMcp).not.toHaveBeenCalled();
		agenticStore.resolvePermission(conversationId, ToolPermissionDecision.ONCE);
		await vi.runAllTimersAsync();
		expect(agenticStore.getPendingPermissionRequest(conversationId)?.toolName).toBe('ask_b');
		expect(executeMcp).not.toHaveBeenCalled();
		agenticStore.resolvePermission(conversationId, ToolPermissionDecision.DENY);
		await vi.runAllTimersAsync();
		await flow;

		expect(executeMcp).toHaveBeenCalledTimes(1);
		expect(executeMcp.mock.calls[0][0].function.name).toBe('ask_a');
		expect(createToolResultMessage).toHaveBeenNthCalledWith(
			2,
			'call-1',
			expect.stringContaining('denied'),
			undefined
		);
		expect(
			onFlowComplete.mock.calls[0][0]?.agentic?.toolCalls?.map((call) => call.success)
		).toEqual([true, false]);
		expect(agenticStore.getPendingPermissionRequest(conversationId)).toBeNull();
	});

	it('treats a repeated MCP server and a non-MCP call as barriers', async () => {
		settingsStore.config.agenticParallelToolCalls = true;
		const pending = Array.from({ length: 6 }, () => deferred<ToolExecutionResult>());
		const started: string[] = [];

		executeMcp.mockImplementation((call) => {
			started.push(call.function.name);

			return pending[Number(call.id.split('-')[1])].promise;
		});
		vi.mocked(ToolsService.executeTool).mockImplementation((name) => {
			started.push(name);

			return pending[3].promise;
		});
		const names = [
			'ask_a',
			'ask_a_followup',
			'ask_b',
			BuiltInTool.SERVER_READ_FILE,
			'ask_c',
			'ask_d'
		];
		const { flow } = start(names);

		await vi.runAllTimersAsync();
		expect(started).toEqual(names.slice(0, 1));

		pending[0].resolve(result('first'));
		await vi.runAllTimersAsync();
		expect(started).toEqual(names.slice(0, 3));
		pending[2].resolve(result('third'));
		await vi.runAllTimersAsync();
		expect(started).toEqual(names.slice(0, 3));
		pending[1].resolve(result('second'));
		await vi.runAllTimersAsync();
		expect(started).toEqual(names.slice(0, 4));
		pending[3].resolve(result('local'));
		await vi.runAllTimersAsync();
		expect(started).toEqual(names);
		pending[4].resolve(result('fifth'));
		pending[5].resolve(result('sixth'));
		await flow;
	});

	it('isolates thrown and reported MCP errors while retaining successful results', async () => {
		settingsStore.config.agenticParallelToolCalls = true;
		const first = deferred<ToolExecutionResult>();

		executeMcp
			.mockImplementationOnce(() => first.promise)
			.mockRejectedValueOnce(new Error('unavailable'))
			.mockResolvedValueOnce(result('remote error', true));
		const { createToolResultMessage, flow, onFlowComplete } = start(['ask_a', 'ask_b', 'ask_c']);

		await vi.runAllTimersAsync();

		expect(executeMcp).toHaveBeenCalledTimes(3);
		expect(sendMessage).toHaveBeenCalledTimes(1);
		first.resolve(result('answer'));
		await flow;
		expect(createToolResultMessage.mock.calls.map(([id, content]) => [id, content])).toEqual([
			['call-0', 'answer'],
			['call-1', 'Error: unavailable'],
			['call-2', 'remote error']
		]);
		expect(
			onFlowComplete.mock.calls[0][0]?.agentic?.toolCalls?.map((call) => call.success)
		).toEqual([true, false, false]);
		expect(sendMessage).toHaveBeenCalledTimes(2);
		expect(agenticStore.getLastError(conversationId)).toBeNull();
	});

	it('propagates cancellation to every call and joins them before ending the flow', async () => {
		settingsStore.config.agenticParallelToolCalls = true;
		const controller = new AbortController();
		const first = deferred<ToolExecutionResult>();
		const second = deferred<ToolExecutionResult>();

		executeMcp
			.mockImplementationOnce(() => first.promise)
			.mockImplementationOnce(() => second.promise);
		const { flow, onFlowComplete } = start(['ask_a', 'ask_b'], {}, controller.signal);

		await vi.runAllTimersAsync();

		expect(executeMcp.mock.calls.map(([, signal]) => signal)).toEqual([
			controller.signal,
			controller.signal
		]);
		controller.abort();
		first.reject(new DOMException('Cancelled', 'AbortError'));
		await vi.runAllTimersAsync();
		expect(agenticStore.getExecutingToolCallIds(conversationId)).toEqual(['call-1']);
		expect(agenticStore.isRunning(conversationId)).toBe(true);
		expect(mcpStore.releaseConnection).not.toHaveBeenCalled();

		second.reject(new DOMException('Cancelled', 'AbortError'));
		await flow;
		expect(agenticStore.getExecutingToolCallIds(conversationId)).toEqual([]);
		expect(agenticStore.isRunning(conversationId)).toBe(false);
		expect(sendMessage).toHaveBeenCalledTimes(1);
		expect(onFlowComplete).toHaveBeenCalledTimes(1);
		expect(mcpStore.releaseConnection).toHaveBeenCalledTimes(1);
	});

	it('keeps a completed result when another call is cancelled', async () => {
		settingsStore.config.agenticParallelToolCalls = true;
		const controller = new AbortController();
		const pending = deferred<ToolExecutionResult>();

		executeMcp
			.mockImplementationOnce(() => pending.promise)
			.mockResolvedValueOnce(result('answer'));
		const { createToolResultMessage, flow, onFlowComplete } = start(
			['ask_a', 'ask_b'],
			{},
			controller.signal
		);

		await vi.runAllTimersAsync();
		expect(createToolResultMessage).not.toHaveBeenCalled();
		controller.abort();
		pending.reject(new DOMException('Cancelled', 'AbortError'));
		await flow;

		expect(createToolResultMessage).toHaveBeenCalledExactlyOnceWith('call-1', 'answer', undefined);
		expect(sendMessage).toHaveBeenCalledTimes(1);
		expect(onFlowComplete).toHaveBeenCalledTimes(1);
		expect(agenticStore.getExecutingToolCallIds(conversationId)).toEqual([]);
	});

	it('cancels a pending approval without launching any calls', async () => {
		settingsStore.config.agenticParallelToolCalls = true;
		vi.mocked(permissionsStore.hasTool).mockReturnValue(false);
		const controller = new AbortController();
		const { flow } = start(['ask_a', 'ask_b'], {}, controller.signal);

		await vi.runAllTimersAsync();
		controller.abort();
		await vi.runAllTimersAsync();
		await flow;

		expect(executeMcp).not.toHaveBeenCalled();
		expect(agenticStore.getPendingPermissionRequest(conversationId)).toBeNull();
		expect(agenticStore.getExecutingToolCallIds(conversationId)).toEqual([]);
		expect(sendMessage).toHaveBeenCalledTimes(1);
	});

	it('honors steering received during approvals before launching the batch', async () => {
		settingsStore.config.agenticParallelToolCalls = true;
		vi.mocked(permissionsStore.hasTool).mockReturnValue(false);
		const { createToolResultMessage, flow } = start(['ask_a', 'ask_b']);

		await vi.runAllTimersAsync();
		agenticStore.resolvePermission(conversationId, ToolPermissionDecision.ONCE);
		await vi.runAllTimersAsync();
		agenticStore.injectSteeringMessage(conversationId, 'Stop and use another approach');
		agenticStore.resolvePermission(conversationId, ToolPermissionDecision.ONCE);
		await vi.runAllTimersAsync();
		await flow;

		expect(executeMcp).not.toHaveBeenCalled();
		expect(createToolResultMessage.mock.calls.map(([id]) => id)).toEqual(['call-0', 'call-1']);
		expect(sendMessage).toHaveBeenCalledTimes(1);
		expect(agenticStore.getPendingSteeringMessageContent(conversationId)).toBe(
			'Stop and use another approach'
		);
	});
});
