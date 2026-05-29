package llm

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestNewClientRequiresAPIKey(t *testing.T) {
	t.Setenv("API_KEY", "")
	t.Setenv("BASE_URL", "https://example.com/v1")

	_, err := NewClient()
	if err == nil {
		t.Fatal("NewClient() error = nil, want missing API key error")
	}
	if !strings.Contains(err.Error(), "API_KEY") {
		t.Fatalf("NewClient() error = %v, want API_KEY mention", err)
	}
}

func TestNewClientRequiresBaseURL(t *testing.T) {
	t.Setenv("API_KEY", "test-key")
	t.Setenv("BASE_URL", "")

	_, err := NewClient()
	if err == nil {
		t.Fatal("NewClient() error = nil, want missing base URL error")
	}
	if !strings.Contains(err.Error(), "BASE_URL") {
		t.Fatalf("NewClient() error = %v, want BASE_URL mention", err)
	}
}

func TestChatReturnsDecodedResponse(t *testing.T) {
	t.Setenv("API_KEY", "test-key")

	var gotReq ChatRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Fatalf("method = %q, want POST", r.Method)
		}
		if r.URL.Path != "/chat/completions" {
			t.Fatalf("path = %q, want /chat/completions", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Fatalf("Authorization = %q, want Bearer test-key", got)
		}
		if got := r.Header.Get("Content-Type"); got != "application/json" {
			t.Fatalf("Content-Type = %q, want application/json", got)
		}

		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("ReadAll() error: %v", err)
		}
		if err := json.Unmarshal(body, &gotReq); err != nil {
			t.Fatalf("Unmarshal request body error: %v", err)
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id": "chatcmpl-test",
			"model": "openai/gpt-4o-mini",
			"choices": [{
				"index": 0,
				"message": {"role": "assistant", "content": "hello"},
				"finish_reason": "stop"
			}]
		}`))
	}))
	t.Cleanup(srv.Close)

	client, err := NewClient(WithBaseURL(srv.URL), WithHTTPClient(srv.Client()))
	if err != nil {
		t.Fatalf("NewClient() unexpected error: %v", err)
	}

	resp, err := client.Chat(context.Background(), ChatRequest{
		Model: "openai/gpt-4o-mini",
		Messages: []Message{
			{Role: "user", Content: "hi"},
		},
	})
	if err != nil {
		t.Fatalf("Chat() unexpected error: %v", err)
	}
	if resp.ID != "chatcmpl-test" {
		t.Fatalf("resp.ID = %q, want chatcmpl-test", resp.ID)
	}
	if len(resp.Choices) != 1 || resp.Choices[0].Message.Content != "hello" {
		t.Fatalf("resp.Choices = %#v, want one choice with content hello", resp.Choices)
	}
	if gotReq.Model != "openai/gpt-4o-mini" {
		t.Fatalf("request model = %q, want openai/gpt-4o-mini", gotReq.Model)
	}
	if len(gotReq.Messages) != 1 || gotReq.Messages[0].Role != "user" || gotReq.Messages[0].Content != "hi" {
		t.Fatalf("request messages = %#v, want one user message hi", gotReq.Messages)
	}
}

func TestGenerateReturnsFirstChoiceContent(t *testing.T) {
	t.Setenv("API_KEY", "test-key")

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id": "chatcmpl-test",
			"model": "anthropic/claude-3.5-sonnet",
			"choices": [{
				"index": 0,
				"message": {"role": "assistant", "content": "reflected instruction"},
				"finish_reason": "stop"
			}]
		}`))
	}))
	t.Cleanup(srv.Close)

	client, err := NewClient(WithBaseURL(srv.URL), WithHTTPClient(srv.Client()))
	if err != nil {
		t.Fatalf("NewClient() unexpected error: %v", err)
	}

	got, err := client.Generate(context.Background(), "anthropic/claude-3.5-sonnet", "improve this prompt")
	if err != nil {
		t.Fatalf("Generate() unexpected error: %v", err)
	}
	if got != "reflected instruction" {
		t.Fatalf("Generate() = %q, want reflected instruction", got)
	}
}

func TestChatSurfacesNon2xxBody(t *testing.T) {
	t.Setenv("API_KEY", "test-key")

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "invalid model", http.StatusBadRequest)
	}))
	t.Cleanup(srv.Close)

	client, err := NewClient(WithBaseURL(srv.URL), WithHTTPClient(srv.Client()))
	if err != nil {
		t.Fatalf("NewClient() unexpected error: %v", err)
	}

	_, err = client.Chat(context.Background(), ChatRequest{
		Model:    "bad/model",
		Messages: []Message{{Role: "user", Content: "hi"}},
	})
	if err == nil {
		t.Fatal("Chat() error = nil, want non-2xx error")
	}
	if !strings.Contains(err.Error(), "400") {
		t.Fatalf("Chat() error = %v, want HTTP 400 mention", err)
	}
	if !strings.Contains(err.Error(), "invalid model") {
		t.Fatalf("Chat() error = %v, want response body mention", err)
	}
}

func TestGenerateRejectsEmptyChoices(t *testing.T) {
	t.Setenv("API_KEY", "test-key")

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-test","model":"openai/gpt-4o-mini","choices":[]}`))
	}))
	t.Cleanup(srv.Close)

	client, err := NewClient(WithBaseURL(srv.URL), WithHTTPClient(srv.Client()))
	if err != nil {
		t.Fatalf("NewClient() unexpected error: %v", err)
	}

	_, err = client.Generate(context.Background(), "openai/gpt-4o-mini", "prompt")
	if err == nil {
		t.Fatal("Generate() error = nil, want empty choices error")
	}
	if !strings.Contains(err.Error(), "choices") {
		t.Fatalf("Generate() error = %v, want choices mention", err)
	}
}

func TestNewClientUsesBaseURLEnv(t *testing.T) {
	t.Setenv("API_KEY", "test-key")

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"}}]}`))
	}))
	t.Cleanup(srv.Close)
	t.Setenv("BASE_URL", srv.URL)

	client, err := NewClient(WithHTTPClient(srv.Client()))
	if err != nil {
		t.Fatalf("NewClient() unexpected error: %v", err)
	}

	got, err := client.Generate(context.Background(), "test/model", "hi")
	if err != nil {
		t.Fatalf("Generate() unexpected error: %v", err)
	}
	if got != "ok" {
		t.Fatalf("Generate() = %q, want ok", got)
	}
}
