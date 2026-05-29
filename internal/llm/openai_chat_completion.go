package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
)

var (
	errMissingAPIKey  = errors.New("llm: API_KEY is not set")
	errMissingBaseURL = errors.New("llm: BASE_URL is not set")
)

// Client calls an OpenAI-compatible /chat/completions endpoint.
type Client struct {
	httpClient *http.Client
	baseURL    string
	apiKey     string
}

type Option func(*Client)

func WithBaseURL(baseURL string) Option {
	return func(c *Client) {
		c.baseURL = strings.TrimRight(baseURL, "/")
	}
}

func WithHTTPClient(httpClient *http.Client) Option {
	return func(c *Client) {
		c.httpClient = httpClient
	}
}

func NewClient(opts ...Option) (*Client, error) {
	apiKey := strings.TrimSpace(os.Getenv("API_KEY"))
	if apiKey == "" {
		return nil, errMissingAPIKey
	}

	baseURL := strings.TrimSpace(os.Getenv("BASE_URL"))

	client := &Client{
		httpClient: http.DefaultClient,
		baseURL:    baseURL,
		apiKey:     apiKey,
	}
	for _, opt := range opts {
		opt(client)
	}
	if strings.TrimSpace(client.baseURL) == "" {
		return nil, errMissingBaseURL
	}
	return client, nil
}

func (c *Client) Chat(ctx context.Context, req ChatRequest) (ChatResponse, error) {
	if err := ctx.Err(); err != nil {
		return ChatResponse{}, err
	}

	body, err := json.Marshal(req)
	if err != nil {
		return ChatResponse{}, fmt.Errorf("llm: marshal chat request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return ChatResponse{}, fmt.Errorf("llm: create chat request: %w", err)
	}
	httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return ChatResponse{}, fmt.Errorf("llm: chat request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return ChatResponse{}, fmt.Errorf("llm: read chat response: %w", err)
	}
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		return ChatResponse{}, fmt.Errorf("llm: chat request failed: status %d: %s", resp.StatusCode, strings.TrimSpace(string(respBody)))
	}

	var out ChatResponse
	if err := json.Unmarshal(respBody, &out); err != nil {
		return ChatResponse{}, fmt.Errorf("llm: decode chat response: %w", err)
	}
	return out, nil
}

func (c *Client) Generate(ctx context.Context, model, prompt string) (string, error) {
	resp, err := c.Chat(ctx, ChatRequest{
		Model: model,
		Messages: []Message{
			{Role: "user", Content: prompt},
		},
	})
	if err != nil {
		return "", err
	}
	if len(resp.Choices) == 0 {
		return "", errors.New("llm: chat response has no choices")
	}
	return resp.Choices[0].Message.Content, nil
}
