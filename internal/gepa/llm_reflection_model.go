package gepa

import (
	"context"
	"errors"

	"github.com/anath2/gepa-go/internal/llm"
)

var errEmptyReflectionResponse = errors.New("gepa: reflection chat response has no choices")

// ReflectionModel generates free-text from a rendered reflection prompt.
type ReflectionModel interface {
	Generate(ctx context.Context, prompt string) (string, error)
}

type llmReflectionModel struct {
	model llm.Model
}

// NewLLMReflectionModel returns a ReflectionModel backed by the given bound LLM model.
func NewLLMReflectionModel(model llm.Model) ReflectionModel {
	return llmReflectionModel{model: model}
}

func (m llmReflectionModel) Generate(ctx context.Context, prompt string) (string, error) {
	if m.model.Client == nil {
		return "", errors.New("gepa: nil model client")
	}
	if err := ctx.Err(); err != nil {
		return "", err
	}

	req := llm.ChatRequest{
		Model: m.model.Name,
		Messages: []llm.Message{
			{Role: "user", Content: prompt},
		},
	}
	if m.model.ReasoningEffort != "" {
		req.Reasoning = map[string]any{"effort": m.model.ReasoningEffort}
	}

	resp, err := m.model.Client.Chat(ctx, req)
	if err != nil {
		return "", err
	}
	if len(resp.Choices) == 0 {
		return "", errEmptyReflectionResponse
	}
	return resp.Choices[0].Message.Content, nil
}
