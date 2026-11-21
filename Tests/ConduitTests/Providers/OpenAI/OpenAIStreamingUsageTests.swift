// OpenAIStreamingUsageTests.swift
// Conduit Tests
//
// Token usage on chat-completions streams: the request asks for it
// (stream_options.include_usage) and the parser reads it from the final
// usage-only chunk ({"choices": [], "usage": {...}}).
//
// Wire shape follows OpenAI's chat-completions streaming reference: with
// include_usage, every chunk carries "usage": null except one extra last chunk
// whose choices array is empty.

#if CONDUIT_TRAIT_OPENAI || CONDUIT_TRAIT_OPENROUTER
import Foundation
import Testing
@testable import ConduitAdvanced

@Suite("OpenAI Streaming Usage Tests")
struct OpenAIStreamingUsageTests {

    private func provider() -> OpenAIProvider {
        OpenAIProvider(configuration: .openAI(apiKey: "sk-test"))
    }

    /// Feeds recorded SSE lines through the chat-completions stream parser.
    private func parse(_ lines: [String]) async throws -> [GenerationChunk] {
        let sut = provider()
        let source = AsyncStream<String> { continuation in
            for line in lines {
                continuation.yield(line)
            }
            continuation.finish()
        }
        let (stream, continuation) = AsyncThrowingStream<GenerationChunk, Error>.makeStream()
        try await sut.processChatCompletionsStream(lines: source, continuation: continuation)
        continuation.finish()

        var chunks: [GenerationChunk] = []
        for try await chunk in stream {
            chunks.append(chunk)
        }
        return chunks
    }

    // MARK: - Request

    @Test("Streaming chat-completions request asks for usage")
    func streamingRequestIncludesUsageOption() throws {
        let body = provider().buildRequestBody(
            messages: [.user("Hi")],
            model: .gpt4o,
            config: .default,
            stream: true
        )
        let options = try #require(body["stream_options"] as? [String: Any])
        #expect(options["include_usage"] as? Bool == true)
    }

    @Test("Non-streaming chat-completions request sends no stream_options")
    func nonStreamingRequestOmitsUsageOption() {
        let body = provider().buildRequestBody(
            messages: [.user("Hi")],
            model: .gpt4o,
            config: .default,
            stream: false
        )
        #expect(body["stream_options"] == nil)
    }

    // MARK: - Parsing

    @Test("Recorded text stream ending in a usage-only chunk yields the usage")
    func textStreamUsageOnlyChunk() async throws {
        let chunks = try await parse([
            #"data: {"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"},"finish_reason":null}],"usage":null}"#,
            "",
            #"data: {"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":null}],"usage":null}"#,
            "",
            #"data: {"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":null}"#,
            "",
            #"data: {"id":"c1","object":"chat.completion.chunk","choices":[],"usage":{"prompt_tokens":12,"completion_tokens":2,"total_tokens":14}}"#,
            "",
            "data: [DONE]",
            "",
        ])

        #expect(chunks.map(\.text).joined() == "Hello")
        #expect(chunks.dropLast().allSatisfy { $0.usage == nil })

        let last = try #require(chunks.last)
        #expect(last.usage == UsageStats(promptTokens: 12, completionTokens: 2))
        #expect(last.isComplete)
        #expect(last.finishReason == .stop)
        #expect(last.text.isEmpty)
        #expect(last.completedToolCalls == nil)
    }

    @Test("Tool-call stream yields the completed calls, then the usage")
    func toolCallStreamUsageAfterToolCalls() async throws {
        let chunks = try await parse([
            #"data: {"choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}],"usage":null}"#,
            "",
            #"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":\"SF\"}"}}]},"finish_reason":null}],"usage":null}"#,
            "",
            #"data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":null}"#,
            "",
            #"data: {"choices":[],"usage":{"prompt_tokens":40,"completion_tokens":9,"total_tokens":49}}"#,
            "",
            "data: [DONE]",
            "",
        ])

        let toolIndex = try #require(chunks.firstIndex { $0.completedToolCalls?.isEmpty == false })
        let usageIndex = try #require(chunks.firstIndex { $0.usage != nil })
        #expect(toolIndex < usageIndex)
        #expect(chunks[toolIndex].completedToolCalls?.first?.toolName == "get_weather")
        #expect(chunks[usageIndex].usage == UsageStats(promptTokens: 40, completionTokens: 9))
        #expect(chunks[usageIndex].finishReason == .toolCalls)
        #expect(chunks[usageIndex].completedToolCalls == nil)
    }

    @Test("Usage sent on the finishing chunk itself is kept on that chunk")
    func usageOnFinishChunk() async throws {
        let chunks = try await parse([
            #"data: {"choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}"#,
            "",
            #"data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}"#,
            "",
            "data: [DONE]",
            "",
        ])

        let usageChunks = chunks.filter { $0.usage != nil }
        #expect(usageChunks.count == 1)
        #expect(usageChunks.first?.finishReason == .stop)
        #expect(usageChunks.first?.usage == UsageStats(promptTokens: 5, completionTokens: 1))
    }

    @Test("A stream without usage still completes and reports none")
    func streamWithoutUsage() async throws {
        let chunks = try await parse([
            #"data: {"choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}"#,
            "",
            #"data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
            "",
            "data: [DONE]",
            "",
        ])

        #expect(chunks.map(\.text).joined() == "Hi")
        #expect(chunks.allSatisfy { $0.usage == nil })
        #expect(chunks.last?.finishReason == .stop)
    }
}
#endif
