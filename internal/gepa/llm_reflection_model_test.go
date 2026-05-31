package gepa

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/anath2/gepa-go/internal/llm"
)

func TestLLMReflectionModelUsesBoundModelName(t *testing.T) {
	t.Setenv("API_KEY", "test-key")

	var gotReq llm.ChatRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotReq); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		body, err := json.Marshal(map[string]any{
			"choices": []map[string]any{
				{"message": map[string]any{"content": "```\nImproved prompt.\n```"}},
			},
		})
		if err != nil {
			t.Fatalf("marshal response: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(body)
	}))
	t.Cleanup(srv.Close)

	client, err := llm.NewClient(llm.WithBaseURL(srv.URL), llm.WithHTTPClient(srv.Client()))
	if err != nil {
		t.Fatalf("NewClient() unexpected error: %v", err)
	}

	model := NewLLMReflectionModel(llm.Model{Name: "reflection-model", Client: client})
	got, err := model.Generate(context.Background(), "improve this prompt")
	if err != nil {
		t.Fatalf("Generate() unexpected error: %v", err)
	}
	if got != "```\nImproved prompt.\n```" {
		t.Fatalf("Generate() = %q, want raw model content", got)
	}
	if gotReq.Model != "reflection-model" {
		t.Fatalf("request model = %q, want reflection-model", gotReq.Model)
	}
}
