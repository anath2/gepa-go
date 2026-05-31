package rollout

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/anath2/gepa-go/internal/config"
	"github.com/anath2/gepa-go/internal/gepa"
	"github.com/anath2/gepa-go/internal/llm"
	"github.com/anath2/gepa-go/internal/program"
)

func TestLLMTaskModelUsesStructuredOutputAndParsesJSON(t *testing.T) {
	t.Setenv("API_KEY", "test-key")

	var gotReq llm.ChatRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotReq); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"{\"answer\":\"Paris\"}"}}]}`))
	}))
	t.Cleanup(srv.Close)

	client, err := llm.NewClient(llm.WithBaseURL(srv.URL), llm.WithHTTPClient(srv.Client()))
	if err != nil {
		t.Fatalf("NewClient() unexpected error: %v", err)
	}

	model := NewLLMTaskModel(llm.Model{Name: "task-model", Client: client})
	resp, err := model.Generate(context.Background(), ModuleRequest{
		ModuleName:  "answer",
		Instruction: "Answer the question.",
		Input:       map[string]any{"question": "capital?"},
		OutputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"answer": map[string]any{"type": "string"},
			},
			"required": []any{"answer"},
		},
	})
	if err != nil {
		t.Fatalf("Generate() unexpected error: %v", err)
	}
	if resp.Output["answer"] != "Paris" {
		t.Fatalf("Output.answer = %#v, want Paris", resp.Output["answer"])
	}
	if gotReq.Model != "task-model" {
		t.Fatalf("request model = %q, want task-model", gotReq.Model)
	}
	if gotReq.ResponseFormat == nil {
		t.Fatal("ResponseFormat = nil, want json_schema")
	}
}

func TestLLMTaskModelEvaluatorEndToEnd(t *testing.T) {
	t.Setenv("API_KEY", "test-key")

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"{\"answer\":\"a1\"}"}}]}`))
	}))
	t.Cleanup(srv.Close)
	t.Setenv("BASE_URL", srv.URL)

	client, err := llm.NewClient(llm.WithHTTPClient(srv.Client()))
	if err != nil {
		t.Fatalf("NewClient() unexpected error: %v", err)
	}

	eval := Evaluator{
		Program: program.Program{
			Modules: []program.Module{
				{
					Name:         "answer",
					InputSchema:  objectSchema("question", program.KindString),
					OutputSchema: objectSchema("answer", program.KindString),
				},
			},
		},
		Config: config.Config{TaskModel: "task-model", Metric: config.Metric{Kind: "exact_match", Field: "answer"}},
		Model:  NewLLMTaskModel(llm.Model{Name: "task-model", Client: client}),
	}

	results, err := eval.Evaluate(context.Background(), gepa.Candidate{"answer": "Answer."}, []program.Example{
		{Input: map[string]any{"question": "q1"}, Expected: map[string]any{"answer": "a1"}},
	})
	if err != nil {
		t.Fatalf("Evaluate() unexpected error: %v", err)
	}
	if len(results) != 1 || results[0].Score != 1 {
		t.Fatalf("results = %#v, want one exact-match score", results)
	}
}
