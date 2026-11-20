// AnthropicProvider+Thinking.swift
// Conduit
//
// Anthropic thinking: the `thinking` request field, streaming accumulation
// state, and replay of signed thinking blocks in assistant turns.

#if CONDUIT_TRAIT_ANTHROPIC
import Foundation

/// Accumulation state for one streamed `thinking` / `redacted_thinking` block.
///
/// Thinking text arrives as `thinking_delta` events and the signature as a
/// single `signature_delta` just before the block's `content_block_stop`.
/// `redacted_thinking` blocks arrive whole in `content_block_start`.
internal struct AnthropicThinkingBlockState: Sendable, Equatable {
    /// Whether this is a `redacted_thinking` block.
    var isRedacted: Bool
    /// Thinking text accumulated so far (empty under `display: "omitted"`).
    var text: String
    /// The block's signature (thinking) or opaque data (redacted_thinking).
    var signature: String?

    init(isRedacted: Bool = false, text: String = "", signature: String? = nil) {
        self.isRedacted = isRedacted
        self.text = text
        self.signature = signature
    }
}

// MARK: - Thinking (request shape and replay)

extension AnthropicProvider {

    /// `ReasoningDetail.format` used for Anthropic thinking blocks.
    static let anthropicReasoningFormat = "anthropic-claude-v1"

    /// `ReasoningDetail.type` for summarized thinking text.
    static let anthropicThinkingDetailType = "reasoning.summary"

    /// `ReasoningDetail.type` for `redacted_thinking` blocks (opaque payload in `signature`).
    static let anthropicRedactedThinkingDetailType = "reasoning.encrypted"

    /// Builds the `thinking` request field from a `ReasoningConfig`.
    ///
    /// - Returns `nil` when reasoning is absent, disabled (`enabled == false`
    ///   or effort `.none`), or not asked for (no `enabled`, `effort` or `maxTokens`).
    /// - Current models get `{type: "adaptive", display: "summarized"}`
    ///   (`display: "omitted"` when `exclude == true`). They reject
    ///   `{type: "enabled", budget_tokens}` with HTTP 400.
    /// - Older models that only know budgeted thinking (Haiku 4.5, the 4.5 /
    ///   4.1 / 4.0 / 3.x family) get `{type: "enabled", budget_tokens: N}`
    ///   with `N = max(1024, reasoning.maxTokens ?? 1024)`, kept below
    ///   `maxTokens`; `nil` when `maxTokens` leaves no room for 1024 tokens.
    static func thinkingRequest(
        for reasoning: ReasoningConfig?,
        modelID: String,
        maxTokens: Int
    ) -> AnthropicMessagesRequest.ThinkingRequest? {
        guard let reasoning else { return nil }
        if reasoning.enabled == false { return nil }
        if reasoning.effort == ReasoningEffort.none { return nil }
        let asked = reasoning.enabled == true || reasoning.effort != nil || reasoning.maxTokens != nil
        guard asked else { return nil }

        if usesBudgetThinking(modelID: modelID) {
            var budget = max(1024, reasoning.maxTokens ?? 1024)
            if budget >= maxTokens {
                budget = maxTokens - 1
            }
            guard budget >= 1024 else { return nil }
            return AnthropicMessagesRequest.ThinkingRequest(type: "enabled", budget_tokens: budget)
        }

        return AnthropicMessagesRequest.ThinkingRequest(
            type: "adaptive",
            display: reasoning.exclude == true ? "omitted" : "summarized"
        )
    }

    /// Whether a model id belongs to the older family that needs
    /// `{type: "enabled", budget_tokens}` instead of adaptive thinking.
    ///
    /// Everything not listed here (Fable, Mythos, Opus 4.6+, Sonnet 4.6+,
    /// Haiku 5.x, Opus/Sonnet 5.x) uses adaptive thinking.
    static func usesBudgetThinking(modelID: String) -> Bool {
        let id = modelID.lowercased()
        let budgetFamilies = [
            "claude-3",
            "claude-haiku-4-5",
            "claude-sonnet-4-5",
            "claude-opus-4-5",
            "claude-opus-4-1",
            "claude-opus-4-0",
            "claude-sonnet-4-0",
            "claude-opus-4-2025",
            "claude-sonnet-4-2025",
        ]
        return budgetFamilies.contains { id.contains($0) }
    }

    /// Turns an assistant message's signed reasoning details back into
    /// Anthropic `thinking` / `redacted_thinking` content parts, unchanged.
    ///
    /// Details without a signature (partial streaming details) and details
    /// from other providers' formats are skipped.
    static func replayedThinkingParts(
        _ details: [ReasoningDetail]?
    ) -> [AnthropicMessagesRequest.MessageContent.ContentPart] {
        guard let details, !details.isEmpty else { return [] }
        return details.compactMap { detail in
            guard detail.format == anthropicReasoningFormat,
                  let signature = detail.signature, !signature.isEmpty else {
                return nil
            }
            if detail.type == anthropicRedactedThinkingDetailType {
                return AnthropicMessagesRequest.MessageContent.ContentPart(
                    type: "redacted_thinking",
                    text: nil,
                    source: nil,
                    data: signature
                )
            }
            return AnthropicMessagesRequest.MessageContent.ContentPart(
                type: "thinking",
                text: nil,
                source: nil,
                thinking: detail.content ?? "",
                signature: signature
            )
        }
    }
}

#endif // CONDUIT_TRAIT_ANTHROPIC
