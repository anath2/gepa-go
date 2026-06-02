package rollout

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/anath2/gepa-go/internal/llm"
)

var (
	errEmptyChatResponse  = errors.New("rollout: chat response has no choices")
	errDecodeModuleOutput = errors.New("rollout: decode module output")
)

type llmTaskModel struct {
	model llm.Model
}

// NewLLMTaskModel returns a TaskModel backed by the given bound LLM model.
func NewLLMTaskModel(model llm.Model) TaskModel {
	return llmTaskModel{model: model}
}

func (m llmTaskModel) Generate(ctx context.Context, req ModuleRequest) (ModuleResponse, error) {
	if m.model.Client == nil {
		return ModuleResponse{}, errors.New("rollout: nil model client")
	}
	if err := ctx.Err(); err != nil {
		return ModuleResponse{}, err
	}

	inputJSON, err := json.Marshal(req.Input)
	if err != nil {
		return ModuleResponse{}, fmt.Errorf("rollout: marshal module input: %w", err)
	}

	prompt := strings.TrimSpace(req.Instruction) + "\n\nInput JSON:\n" + string(inputJSON)
	chatReq := llm.ChatRequest{
		Model: m.model.Name,
		Messages: []llm.Message{
			{Role: "user", Content: prompt},
		},
	}
	if m.model.ReasoningEffort != "" {
		chatReq.Reasoning = map[string]any{"effort": m.model.ReasoningEffort}
	}
	if req.OutputSchema != nil {
		chatReq.ResponseFormat = map[string]any{
			"type": "json_schema",
			"json_schema": map[string]any{
				"name":   "module_output",
				"strict": true,
				"schema": req.OutputSchema,
			},
		}
	}

	resp, err := m.model.Client.Chat(ctx, chatReq)
	if err != nil {
		return ModuleResponse{}, err
	}
	if len(resp.Choices) == 0 {
		return ModuleResponse{}, errEmptyChatResponse
	}

	content := strings.TrimSpace(resp.Choices[0].Message.Content)
	var output map[string]any
	if err := json.Unmarshal([]byte(content), &output); err != nil {
		return ModuleResponse{}, fmt.Errorf("%w: %w", errDecodeModuleOutput, err)
	}
	return ModuleResponse{Output: output}, nil
}
