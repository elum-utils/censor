package ai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/elum-utils/censor/models"
)

func TestNewDeepSeekAdapterValidationAndDefaults(t *testing.T) {
	if _, err := NewDeepSeekAdapter(DeepSeekOptions{}); err == nil {
		t.Fatalf("expected error")
	}
	a, err := NewDeepSeekAdapter(DeepSeekOptions{APIKey: "k"})
	if err != nil {
		t.Fatal(err)
	}
	if a.model == "" || a.baseURL == "" {
		t.Fatalf("defaults are not set")
	}
}

func TestNewDeepSeekAdapterSystemPromptPriority(t *testing.T) {
	a, err := NewDeepSeekAdapter(DeepSeekOptions{
		APIKey:       "k",
		SystemPrompt: "prompt-A",
		SystemHint:   "prompt-B",
	})
	if err != nil {
		t.Fatal(err)
	}
	if a.prompt != "prompt-A" {
		t.Fatalf("expected SystemPrompt to have priority, got: %s", a.prompt)
	}
	if got := a.systemPromptFor(true); got != "prompt-A" {
		t.Fatalf("custom prompt should be used as-is, got: %s", got)
	}
}

func TestSystemPromptForMode(t *testing.T) {
	a, err := NewDeepSeekAdapter(DeepSeekOptions{APIKey: "k"})
	if err != nil {
		t.Fatal(err)
	}
	single := a.systemPromptFor(false)
	batch := a.systemPromptFor(true)
	if single == batch {
		t.Fatalf("single and batch prompts must differ")
	}
	if !strings.Contains(single, "{\"a\":status_code") {
		t.Fatalf("single prompt format not found")
	}
	if !strings.Contains(batch, "[{\"a\":status_code") {
		t.Fatalf("batch prompt format not found")
	}
}

func TestExtractContentErrorsAndFence(t *testing.T) {
	if _, err := extractContent([]byte(`{"choices":[]}`)); err == nil {
		t.Fatalf("expected error")
	}
	body := []byte("{\"choices\":[{\"message\":{\"content\":\"```json {\\\"a\\\":1,\\\"b\\\":\\\"ok\\\",\\\"c\\\":1,\\\"d\\\":[]} ```\"}}]}")
	content, err := extractContent(body)
	if err != nil {
		t.Fatal(err)
	}
	if content == "" {
		t.Fatalf("expected content")
	}
}

func TestParseResultsSingleInvalidCode(t *testing.T) {
	out, err := parseResults(`{"a":9,"b":"x","c":0.1,"d":[]}`)
	if err != nil {
		t.Fatal(err)
	}
	if out[0].StatusCode != models.StatusHumanReview {
		t.Fatalf("expected fallback status (human review)")
	}
}

func TestAnalyzeBatchHTTP(t *testing.T) {
	a, err := NewDeepSeekAdapter(DeepSeekOptions{APIKey: "k", BaseURL: "http://x", Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	a.client.SetTransport(roundTripFunc(func(r *http.Request) (*http.Response, error) {
		if r.Method != http.MethodPost {
			t.Fatalf("unexpected method")
		}
		if r.URL.Path != "/chat/completions" {
			t.Fatalf("unexpected endpoint: %s", r.URL.Path)
		}

		var payload map[string]any
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if stream, ok := payload["stream"].(bool); !ok || stream {
			t.Fatalf("expected stream=false")
		}
		rf, ok := payload["response_format"].(map[string]any)
		if !ok || rf["type"] != "json_object" {
			t.Fatalf("expected response_format json_object")
		}

		body := `{"choices":[{"message":{"content":"[{\"a\":2,\"b\":\"abuse\",\"c\":0.8,\"d\":[\"bad\"],\"e\":2,\"f\":1}]"}}]}`
		return &http.Response{
			StatusCode: 200,
			Header:     make(http.Header),
			Body:       io.NopCloser(strings.NewReader(body)),
		}, nil
	}))
	res, err := a.AnalyzeBatch(context.Background(), []models.Message{{ID: 1, User: 2, Data: "bad"}})
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 1 || res[0].StatusCode != models.StatusNonCriticalAbuse {
		t.Fatalf("unexpected result: %+v", res)
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (r roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return r(req)
}
