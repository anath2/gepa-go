package llm

import "context"

// ChatCompleter is the transport interface for OpenAI-compatible chat completion.
type ChatCompleter interface {
	Chat(ctx context.Context, req ChatRequest) (ChatResponse, error)
}

// Model binds a model name and optional provider controls to a chat completion client.
type Model struct {
	Name            string
	ReasoningEffort string
	Client          ChatCompleter
}
