package ai

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/elum-utils/censor/models"
)

func TestNameAndAnalyze(t *testing.T) {
	a, err := NewDeepSeekAdapter(DeepSeekOptions{APIKey: "k", BaseURL: "http://x", Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	if a.Name() != "deepseek" {
		t.Fatalf("unexpected name")
	}
	a.client.SetTransport(roundTripFunc(func(*http.Request) (*http.Response, error) {
		body := `{"choices":[{"message":{"content":"{\"a\":1,\"b\":\"ok\",\"c\":0.9,\"d\":[]}"}}]}`
		return &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader(body)), Header: make(http.Header)}, nil
	}))
	res, err := a.Analyze(context.Background(), models.Message{ID: 1, User: 2, Data: "x"})
	if err != nil || res.StatusCode != models.StatusClean {
		t.Fatalf("unexpected analyze: res=%+v err=%v", res, err)
	}
}

func TestAnalyzeBatchBranches(t *testing.T) {
	a, err := NewDeepSeekAdapter(DeepSeekOptions{APIKey: "k", BaseURL: "http://x", Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	res, err := a.AnalyzeBatch(context.Background(), nil)
	if err != nil || len(res) != 0 {
		t.Fatalf("expected empty")
	}

	a.client.SetTransport(roundTripFunc(func(*http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: 500, Body: io.NopCloser(strings.NewReader("boom")), Header: make(http.Header)}, nil
	}))
	if _, err := a.AnalyzeBatch(context.Background(), []models.Message{{ID: 1, User: 1, Data: "x"}}); err == nil {
		t.Fatalf("expected status error")
	}

	a.client.SetTransport(roundTripFunc(func(*http.Request) (*http.Response, error) {
		body := `{"choices":[{"message":{"content":"{\"a\":2,\"b\":\"abuse\",\"c\":0.8,\"d\":[\"bad\"]}"}}]}`
		return &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader(body)), Header: make(http.Header)}, nil
	}))
	out, err := a.AnalyzeBatch(context.Background(), []models.Message{{ID: 1, User: 11, Data: "a"}, {ID: 2, User: 22, Data: "b"}})
	if err != nil || len(out) != 2 {
		t.Fatalf("unexpected fanout: out=%+v err=%v", out, err)
	}
}

func TestAnalyzeBatchDoError(t *testing.T) {
	a, err := NewDeepSeekAdapter(DeepSeekOptions{APIKey: "k", BaseURL: "http://x", Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	a.client.SetTransport(roundTripFunc(func(*http.Request) (*http.Response, error) {
		return nil, errors.New("dial")
	}))
	if _, err := a.AnalyzeBatch(context.Background(), []models.Message{{ID: 1, User: 1, Data: "x"}}); err == nil {
		t.Fatalf("expected transport error")
	}
}

func TestAlignResultsFallbackOrder(t *testing.T) {
	msgs := []models.Message{{ID: 10, User: 2}, {ID: 20, User: 3}}
	in := []models.AIResult{{StatusCode: models.StatusClean}, {StatusCode: models.StatusCritical}}
	out := alignResults(msgs, in)
	if out[0].MessageID != 10 || out[1].MessageID != 20 {
		t.Fatalf("unexpected order: %+v", out)
	}
}

func TestExtractAndParseErrors(t *testing.T) {
	if _, err := extractContent([]byte(`{"choices":[{"message":{"content":""}}]}`)); err == nil {
		t.Fatalf("expected empty content error")
	}
	if _, err := parseResults(""); err == nil {
		t.Fatalf("expected parse error")
	}
	arr, err := parseResults(`[{"a":8,"b":"x","c":0.1,"d":[]}]`)
	if err != nil {
		t.Fatal(err)
	}
	if arr[0].StatusCode != models.StatusSuspicious {
		t.Fatalf("expected fallback suspicious")
	}
}

func TestBuildChatCompletionsURL(t *testing.T) {
	cases := map[string]string{
		"":                                "https://api.deepseek.com/chat/completions",
		"https://api.deepseek.com":        "https://api.deepseek.com/chat/completions",
		"https://api.deepseek.com/v1":     "https://api.deepseek.com/v1/chat/completions",
		"https://api.deepseek.com/custom": "https://api.deepseek.com/custom/chat/completions",
		"https://api.deepseek.com/chat/completions": "https://api.deepseek.com/chat/completions",
	}
	for in, want := range cases {
		if got := buildChatCompletionsURL(in); got != want {
			t.Fatalf("url mismatch: in=%s got=%s want=%s", in, got, want)
		}
	}
}
