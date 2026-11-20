// AnthropicThinkingTests.swift
// Conduit Tests
//
// Anthropic thinking: request shape (adaptive vs budget), streamed
// thinking_delta / signature_delta decoding, non-streaming parse, and replay
// of signed thinking blocks in assistant turns.
//
// Wire shapes follow Anthropic's streaming docs
// (platform.claude.com/docs/en/build-with-claude/streaming): a thinking block
// opens with content_block_start {"type":"thinking","thinking":"","signature":""},
// streams thinking_delta {"thinking": ...}, then one signature_delta
// {"signature": ...} just before content_block_stop.

#if CONDUIT_TRAIT_ANTHROPIC
import Foundation
import Testing
@testable import ConduitAdvanced

@Suite("Anthropic thinking")
struct AnthropicThinkingTests {

    private func provider() -> AnthropicProvider {
        AnthropicProvider(configuration: .standard(apiKey: "sk-test-only"))
    }

    private func requestJSON(
        messages: [Message],
        model: String,
        config: GenerateConfig
    ) async throws -> [String: Any] {
        let request = try await provider().buildRequestBody(
            messages: messages,
            model: .anthropic(model),
            config: config,
            stream: true
        )
        let data = try JSONEncoder().encode(request)
        return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    // MARK: - Request shape

    @Test("No reasoning config sends no thinking field and keeps temperature")
    func noReasoningNoThinking() async throws {
        let json = try await requestJSON(
            messages: [.user("hi")],
            model: "claude-fable-5",
            config: .default
        )
        #expect(json["thinking"] == nil)
        #expect(json["temperature"] != nil)
    }

    @Test("Current models get adaptive thinking with summarized display and no sampling params")
    func adaptiveSummarized() async throws {
        let config = GenerateConfig.default
            .maxTokens(8000)
            .reasoning(ReasoningConfig(enabled: true))
        for model in ["claude-fable-5", "claude-opus-4-8", "claude-sonnet-5", "claude-opus-4-6", "claude-sonnet-4-6"] {
            let json = try await requestJSON(messages: [.user("hi")], model: model, config: config)
            let thinking = try #require(json["thinking"] as? [String: Any], "\(model)")
            #expect(thinking["type"] as? String == "adaptive", "\(model)")
            #expect(thinking["display"] as? String == "summarized", "\(model)")
            #expect(thinking["budget_tokens"] == nil, "\(model)")
            #expect(json["temperature"] == nil, "\(model)")
            #expect(json["top_p"] == nil, "\(model)")
            #expect(json["top_k"] == nil, "\(model)")
        }
    }

    @Test("Effort alone turns thinking on; exclude asks for omitted display")
    func effortAndExclude() async throws {
        let onlyEffort = try await requestJSON(
            messages: [.user("hi")],
            model: "claude-fable-5",
            config: GenerateConfig.default.reasoning(.high)
        )
        let thinking = try #require(onlyEffort["thinking"] as? [String: Any])
        #expect(thinking["type"] as? String == "adaptive")

        let excluded = try await requestJSON(
            messages: [.user("hi")],
            model: "claude-fable-5",
            config: GenerateConfig.default.reasoning(ReasoningConfig(effort: .medium, exclude: true))
        )
        let excludedThinking = try #require(excluded["thinking"] as? [String: Any])
        #expect(excludedThinking["display"] as? String == "omitted")
    }

    @Test("Disabled reasoning sends no thinking")
    func disabledReasoning() async throws {
        for reasoning in [
            ReasoningConfig(enabled: false),
            ReasoningConfig(effort: ReasoningEffort.none),
            ReasoningConfig(),
        ] {
            let json = try await requestJSON(
                messages: [.user("hi")],
                model: "claude-fable-5",
                config: GenerateConfig.default.reasoning(reasoning)
            )
            #expect(json["thinking"] == nil)
        }
    }

    @Test("Budget-only models get enabled thinking with a budget below max_tokens")
    func budgetModels() async throws {
        let config = GenerateConfig.default
            .maxTokens(8000)
            .reasoning(ReasoningConfig(maxTokens: 2048, enabled: true))
        for model in ["claude-haiku-4-5", "claude-haiku-4-5-20251001", "claude-sonnet-4-5-20250929", "claude-3-7-sonnet-20250219"] {
            let json = try await requestJSON(messages: [.user("hi")], model: model, config: config)
            let thinking = try #require(json["thinking"] as? [String: Any], "\(model)")
            #expect(thinking["type"] as? String == "enabled", "\(model)")
            #expect(thinking["budget_tokens"] as? Int == 2048, "\(model)")
            #expect(thinking["display"] == nil, "\(model)")
        }

        // A small reasoning budget is raised to the API minimum of 1024.
        let small = try await requestJSON(
            messages: [.user("hi")],
            model: "claude-haiku-4-5",
            config: GenerateConfig.default.maxTokens(4000).reasoning(ReasoningConfig(maxTokens: 100, enabled: true))
        )
        let smallThinking = try #require(small["thinking"] as? [String: Any])
        #expect(smallThinking["budget_tokens"] as? Int == 1024)

        // No room for 1024 thinking tokens below max_tokens: no thinking.
        let noRoom = try await requestJSON(
            messages: [.user("hi")],
            model: "claude-haiku-4-5",
            config: GenerateConfig.default.maxTokens(1000).reasoning(ReasoningConfig(enabled: true))
        )
        #expect(noRoom["thinking"] == nil)
    }

    @Test("An explicit thinkingConfig keeps its budget request")
    func explicitThinkingConfigUnchanged() async throws {
        var configuration = AnthropicConfiguration.standard(apiKey: "sk-test-only")
        configuration.thinkingConfig = ThinkingConfiguration(enabled: true, budgetTokens: 2000)
        let request = try await AnthropicProvider(configuration: configuration).buildRequestBody(
            messages: [.user("hi")],
            model: .anthropic("claude-fable-5"),
            config: GenerateConfig.default.reasoning(ReasoningConfig(enabled: true))
        )
        #expect(request.thinking == AnthropicMessagesRequest.ThinkingRequest(type: "enabled", budget_tokens: 2000))
    }

    // MARK: - Replay

    @Test("Assistant turn replays the signed thinking block before text and tool_use")
    func assistantReplay() async throws {
        let toolCall = try Transcript.ToolCall(id: "toolu_1", toolName: "lookup", argumentsJSON: #"{"q":"x"}"#)
        let assistant = Message(
            role: .assistant,
            content: .text("Let me look."),
            metadata: MessageMetadata(
                toolCalls: [toolCall],
                reasoningDetails: [
                    // Partial (unsigned) detail: must not be replayed.
                    ReasoningDetail(id: "0", type: "reasoning.summary", format: "anthropic-claude-v1",
                                    index: 0, content: "I should"),
                    ReasoningDetail(id: "0", type: "reasoning.summary", format: "anthropic-claude-v1",
                                    index: 0, content: "I should look it up.", signature: "sig-abc"),
                    ReasoningDetail(id: "1", type: "reasoning.encrypted", format: "anthropic-claude-v1",
                                    index: 1, content: nil, signature: "opaque-data"),
                    // Another provider's format: must not be replayed.
                    ReasoningDetail(id: "x", type: "reasoning.text", format: "openai-responses-v1",
                                    index: 0, content: "other", signature: "other-sig"),
                ]
            )
        )
        let toolResult = Message(
            role: .tool,
            content: .text("found"),
            metadata: MessageMetadata(custom: ["tool_call_id": "toolu_1"])
        )
        let json = try await requestJSON(
            messages: [.user("find x"), assistant, toolResult],
            model: "claude-fable-5",
            config: GenerateConfig.default.reasoning(ReasoningConfig(enabled: true))
        )
        let messages = try #require(json["messages"] as? [[String: Any]])
        #expect(messages.count == 3)
        let parts = try #require(messages[1]["content"] as? [[String: Any]])
        #expect(parts.map { $0["type"] as? String } == ["thinking", "redacted_thinking", "text", "tool_use"])
        #expect(parts[0]["thinking"] as? String == "I should look it up.")
        #expect(parts[0]["signature"] as? String == "sig-abc")
        #expect(parts[1]["data"] as? String == "opaque-data")
    }

    @Test("Empty thinking text (omitted display) is replayed as an empty string")
    func emptyThinkingReplayed() {
        let parts = AnthropicProvider.replayedThinkingParts([
            ReasoningDetail(id: "0", type: "reasoning.summary", format: "anthropic-claude-v1",
                            index: 0, content: "", signature: "sig")
        ])
        #expect(parts.count == 1)
        #expect(parts.first?.thinking == "")
        #expect(parts.first?.signature == "sig")
    }

    // MARK: - Streaming

    @Test("Recorded thinking stream yields partial text, then one signed block")
    func streamedThinking() async throws {
        let sut = provider()
        let lines = [
            #"{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}"#,
            #"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"I need to find the GCD."}}"#,
            #"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"\nIt is 21."}}"#,
            #"{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"EqQBCgIYAhIM"}}"#,
            #"{"type":"content_block_stop","index":0}"#,
            #"{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}"#,
            #"{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"21"}}"#,
            #"{"type":"content_block_stop","index":1}"#,
        ]

        var totalTokens = 0
        var activeToolCalls: [Int: (id: String, name: String, jsonBuffer: String)] = [:]
        var completedToolCalls: [Transcript.ToolCall] = []
        var thinkingBlocks: [Int: AnthropicThinkingBlockState] = [:]
        var chunks: [GenerationChunk] = []

        for line in lines {
            let parsed = try await sut.parseStreamEvent(from: Data(line.utf8))
            let event = try #require(parsed)
            if let chunk = try await sut.processStreamEvent(
                event,
                startTime: Date(),
                totalTokens: &totalTokens,
                activeToolCalls: &activeToolCalls,
                completedToolCalls: &completedToolCalls,
                activeThinkingBlocks: &thinkingBlocks
            ) {
                chunks.append(chunk)
            }
        }

        #expect(chunks.count == 4)
        #expect(chunks[0].text.isEmpty)
        #expect(chunks[0].reasoningDetails?.first?.content == "I need to find the GCD.")
        #expect(chunks[0].reasoningDetails?.first?.signature == nil)
        #expect(chunks[1].reasoningDetails?.first?.content == "\nIt is 21.")

        let finished = try #require(chunks[2].reasoningDetails?.first)
        #expect(finished.content == "I need to find the GCD.\nIt is 21.")
        #expect(finished.signature == "EqQBCgIYAhIM")
        #expect(finished.format == "anthropic-claude-v1")
        #expect(finished.type == "reasoning.summary")
        #expect(finished.index == 0)

        #expect(chunks[3].text == "21")
        #expect(chunks[3].reasoningDetails == nil)
        #expect(thinkingBlocks.isEmpty)
    }

    @Test("Omitted display streams no partial text but still yields the signed block")
    func streamedOmittedThinking() async throws {
        let sut = provider()
        let lines = [
            #"{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}"#,
            #"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":""}}"#,
            #"{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig-only"}}"#,
            #"{"type":"content_block_stop","index":0}"#,
        ]
        var totalTokens = 0
        var activeToolCalls: [Int: (id: String, name: String, jsonBuffer: String)] = [:]
        var completedToolCalls: [Transcript.ToolCall] = []
        var thinkingBlocks: [Int: AnthropicThinkingBlockState] = [:]
        var chunks: [GenerationChunk] = []
        for line in lines {
            let parsed = try await sut.parseStreamEvent(from: Data(line.utf8))
            let event = try #require(parsed)
            if let chunk = try await sut.processStreamEvent(
                event,
                startTime: Date(),
                totalTokens: &totalTokens,
                activeToolCalls: &activeToolCalls,
                completedToolCalls: &completedToolCalls,
                activeThinkingBlocks: &thinkingBlocks
            ) {
                chunks.append(chunk)
            }
        }
        #expect(chunks.count == 1)
        #expect(chunks[0].reasoningDetails?.first?.content == "")
        #expect(chunks[0].reasoningDetails?.first?.signature == "sig-only")
    }

    // MARK: - Non-streaming

    @Test("Non-streaming response keeps thinking out of text and in reasoningDetails")
    func nonStreamingParse() async throws {
        let body = """
        {
          "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-fable-5",
          "stop_reason": "end_turn",
          "usage": {"input_tokens": 10, "output_tokens": 20},
          "content": [
            {"type": "thinking", "thinking": "Short summary.", "signature": "sig-1"},
            {"type": "redacted_thinking", "data": "opaque"},
            {"type": "text", "text": "Answer."}
          ]
        }
        """
        let response = try JSONDecoder().decode(AnthropicMessagesResponse.self, from: Data(body.utf8))
        let result = try await provider().convertToGenerationResult(response, startTime: Date())
        #expect(result.text == "Answer.")
        #expect(result.reasoningDetails.count == 2)
        #expect(result.reasoningDetails[0].content == "Short summary.")
        #expect(result.reasoningDetails[0].signature == "sig-1")
        #expect(result.reasoningDetails[1].type == "reasoning.encrypted")
        #expect(result.reasoningDetails[1].signature == "opaque")
    }
}
#endif
