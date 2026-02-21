package ai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/elum-utils/censor/models"
	"github.com/go-resty/resty/v2"
)

const defaultSystemPrompt = `You are a content safety classifier for an anonymous messenger. Return strict JSON only.
Use code:
1 clean
2 non-critical abuse
3 suspicious
4 commercial/off-platform
5 dangerous/illegal
6 critical

Domain rules:
- Intimate/sexual conversation between users is allowed by default.
- Critical priority: detect sales/commercial intent, including intimate services/content sold for money.
- Escalate commercial behavior when messages contain calls to move to other platforms specifically to continue selling, payment, booking, or deal execution.
- Neutral contact exchange is allowed (e.g., "let's chat later in Telegram/VK") when there is no sales intent.
- Base classification on intent and context, not platform names alone.

Trigger tokens can be single words or short phrases, each max 255 characters.
Return compact format: {"a":status_code,"b":"reason","c":confidence,"d":["trigger_tokens"],"e":violator_user_id,"f":message_id}.
For batch input, return array of objects.`

// DeepSeekAdapter is an HTTP AI adapter compatible with OpenAI-style chat completions.
type DeepSeekAdapter struct {
	baseURL  string
	model    string
	client   *resty.Client
	prompt   string
	endpoint string
}

// DeepSeekOptions configures adapter.
type DeepSeekOptions struct {
	APIKey       string
	BaseURL      string
	Model        string
	Timeout      time.Duration
	SystemPrompt string
	// SystemHint is kept for backward compatibility. SystemPrompt has priority.
	SystemHint string
}

// NewDeepSeekAdapter creates adapter instance.
func NewDeepSeekAdapter(opt DeepSeekOptions) (*DeepSeekAdapter, error) {
	if strings.TrimSpace(opt.APIKey) == "" {
		return nil, errors.New("ai: API key is required")
	}
	if strings.TrimSpace(opt.BaseURL) == "" {
		opt.BaseURL = "https://api.deepseek.com"
	}
	if strings.TrimSpace(opt.Model) == "" {
		opt.Model = "deepseek-chat"
	}
	if opt.Timeout <= 0 {
		opt.Timeout = 15 * time.Second
	}
	prompt := defaultSystemPrompt
	if strings.TrimSpace(opt.SystemPrompt) != "" {
		prompt = opt.SystemPrompt
	} else if strings.TrimSpace(opt.SystemHint) != "" {
		prompt = opt.SystemHint
	}
	return &DeepSeekAdapter{
		baseURL:  strings.TrimRight(opt.BaseURL, "/"),
		model:    opt.Model,
		endpoint: buildChatCompletionsURL(strings.TrimRight(opt.BaseURL, "/")),
		client: resty.New().
			SetTimeout(opt.Timeout).
			SetBaseURL(strings.TrimRight(opt.BaseURL, "/")).
			SetAuthToken(opt.APIKey).
			SetHeader("Content-Type", "application/json"),
		prompt: prompt,
	}, nil
}

func (d *DeepSeekAdapter) Name() string { return "deepseek" }

func (d *DeepSeekAdapter) Analyze(ctx context.Context, message models.Message) (models.AIResult, error) {
	results, err := d.AnalyzeBatch(ctx, []models.Message{message})
	if err != nil {
		return models.AIResult{}, err
	}
	if len(results) == 0 {
		return models.AIResult{}, errors.New("ai: empty response")
	}
	return results[0], nil
}

func (d *DeepSeekAdapter) AnalyzeBatch(ctx context.Context, messages []models.Message) ([]models.AIResult, error) {
	if len(messages) == 0 {
		return nil, nil
	}
	payload, err := d.buildPayload(messages)
	if err != nil {
		return nil, err
	}

	resp, err := d.client.R().
		SetContext(ctx).
		SetBody(payload).
		Post(d.endpoint)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode() >= http.StatusMultipleChoices {
		return nil, fmt.Errorf("ai: status %d: %s", resp.StatusCode(), resp.String())
	}

	content, err := extractContent(resp.Body())
	if err != nil {
		return nil, err
	}

	results, err := parseResults(content)
	if err != nil {
		return nil, err
	}
	if len(results) == 1 && len(messages) > 1 {
		for i := range messages {
			copyRes := results[0]
			copyRes.MessageID = messages[i].ID
			if copyRes.ViolatorUserID == 0 {
				copyRes.ViolatorUserID = messages[i].User
			}
			results = append(results, copyRes)
		}
		results = results[1:]
	}
	return alignResults(messages, results), nil
}

func (d *DeepSeekAdapter) buildPayload(messages []models.Message) ([]byte, error) {
	type inputMessage struct {
		ID   int64  `json:"id"`
		User int64  `json:"user"`
		Data string `json:"data"`
	}
	type responseFormat struct {
		Type string `json:"type"`
	}
	type requestMessage struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}
	type requestPayload struct {
		Model          string           `json:"model"`
		Messages       []requestMessage `json:"messages"`
		Temperature    float64          `json:"temperature"`
		Stream         bool             `json:"stream"`
		ResponseFormat responseFormat   `json:"response_format"`
	}
	in := make([]inputMessage, 0, len(messages))
	for _, msg := range messages {
		in = append(in, inputMessage{ID: msg.ID, User: msg.User, Data: msg.Data})
	}
	userPayload, err := json.Marshal(in)
	if err != nil {
		return nil, err
	}

	body := requestPayload{
		Model: d.model,
		Messages: []requestMessage{
			{Role: "system", Content: d.prompt},
			{Role: "user", Content: string(userPayload)},
		},
		Temperature: 0,
		Stream:      false,
		ResponseFormat: responseFormat{
			Type: "json_object",
		},
	}
	return json.Marshal(body)
}

type chatCompletionResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
}

func extractContent(body []byte) (string, error) {
	var resp chatCompletionResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return "", err
	}
	if len(resp.Choices) == 0 {
		return "", errors.New("ai: choices is empty")
	}
	content := strings.TrimSpace(resp.Choices[0].Message.Content)
	if content == "" {
		return "", errors.New("ai: response content is empty")
	}
	content = strings.TrimPrefix(content, "```json")
	content = strings.TrimPrefix(content, "```")
	content = strings.TrimSuffix(content, "```")
	return strings.TrimSpace(content), nil
}

func parseResults(content string) ([]models.AIResult, error) {
	content = strings.TrimSpace(content)
	if content == "" {
		return nil, errors.New("ai: empty result payload")
	}

	if strings.HasPrefix(content, "[") {
		var arr []models.AIResult
		if err := json.Unmarshal([]byte(content), &arr); err != nil {
			return nil, err
		}
		for i := range arr {
			if !arr[i].StatusCode.Valid() {
				arr[i].StatusCode = models.StatusSuspicious
			}
		}
		return arr, nil
	}

	var one models.AIResult
	if err := json.Unmarshal([]byte(content), &one); err != nil {
		return nil, err
	}
	if !one.StatusCode.Valid() {
		one.StatusCode = models.StatusSuspicious
	}
	return []models.AIResult{one}, nil
}

func alignResults(messages []models.Message, results []models.AIResult) []models.AIResult {
	if len(results) == 0 {
		return nil
	}
	byID := make(map[int64]models.AIResult, len(results))
	for _, r := range results {
		if r.MessageID != 0 {
			byID[r.MessageID] = r
		}
	}

	out := make([]models.AIResult, 0, len(messages))
	if len(byID) > 0 {
		for _, msg := range messages {
			res, ok := byID[msg.ID]
			if !ok {
				continue
			}
			if res.ViolatorUserID == 0 {
				res.ViolatorUserID = msg.User
			}
			if res.MessageID == 0 {
				res.MessageID = msg.ID
			}
			out = append(out, res)
		}
		if len(out) > 0 {
			return out
		}
	}

	for i, msg := range messages {
		if i >= len(results) {
			break
		}
		res := results[i]
		if res.ViolatorUserID == 0 {
			res.ViolatorUserID = msg.User
		}
		if res.MessageID == 0 {
			res.MessageID = msg.ID
		}
		out = append(out, res)
	}
	return out
}

func buildChatCompletionsURL(base string) string {
	if base == "" {
		return "https://api.deepseek.com/chat/completions"
	}
	u, err := url.Parse(base)
	if err != nil {
		return strings.TrimRight(base, "/") + "/chat/completions"
	}
	u.Path = strings.TrimRight(u.Path, "/")
	switch u.Path {
	case "":
		u.Path = "/chat/completions"
	case "/v1":
		u.Path = "/v1/chat/completions"
	case "/chat/completions", "/v1/chat/completions":
		// keep as is
	default:
		u.Path = u.Path + "/chat/completions"
	}
	return u.String()
}
