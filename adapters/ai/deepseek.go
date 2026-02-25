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

const defaultSystemPromptBase = `
Classify messages for an anonymous messenger.
Return JSON only.

Priority (highest first): 6 > 5 > 4 > 3 > 2 > 1.
If multiple levels match — return the highest priority.

CRITICAL RULE:
Only classify at a level if there is sufficient explicit evidence.
If context is insufficient for a higher level — downgrade to the highest level that is clearly supported.
Never assume hidden intent.
Never infer payment, bypass, or danger without clear signals.

Codes:
1 clean
2 non-critical abuse (insults, rude language, explicit sexual wording)
3 human review required (ambiguous / insufficient context)
4 suspicious competitor bypass
5 commercial / selling / paid off-platform
6 dangerous / illegal (extreme only)

GENERAL PRINCIPLES:
- Sexual and intimate conversations between consenting adults are fully allowed.
- Explicit sexual content alone is NOT a violation.
- Neutral contact exchange (Telegram, VK, Instagram, etc.) is allowed.
- Detect intent, not keywords alone.
- For levels 1-3 omit trigger tokens.
- For levels 4-6 include short trigger words/phrases (max 255 chars each).

--------------------------------
LEVEL 6 — DANGEROUS / ILLEGAL (EXTREME ONLY)
Use ONLY for:
- Suicide encouragement
- Self-harm instructions
- Real-world violence threats
- Weapons/drug trafficking
- Sexual exploitation of minors
- Terrorism
- Serious criminal activity

Do NOT use 6 without explicit harmful intent.

--------------------------------
LEVEL 5 — COMMERCIAL / PAID
Use only when there is clear payment or selling intent:
- Selling services or products
- Paid intimate content
- Subscriptions / premium access
- Escorting / sugar arrangements
- “Prices”, “buy”, “pay”, “send money”
- Redirecting to another app specifically for payment

If intimate + Telegram + payment intent → 5.
If payment intent is unclear → do NOT use 5.

--------------------------------
LEVEL 4 — SUSPICIOUS COMPETITOR BYPASS
Use ONLY if user:
- Promotes another platform as better/safer
- Encourages leaving because this platform is worse
- Mentions bypassing moderation
- Undermines this service explicitly

Do NOT use 4 if:
- Neutral contact exchange
- “Let's continue in TG” without criticism
- Sharing username only
- No platform comparison
- No bypass intent

--------------------------------
LEVEL 3 — HUMAN REVIEW
Use when:
- Intent is ambiguous
- Possible payment but unclear
- Possible bypass but unclear
- Context missing for reliable classification

--------------------------------
LEVEL 2 — NON-CRITICAL ABUSE
Insults, harassment, rude tone, explicit wording without real threat.

--------------------------------
LEVEL 1 — CLEAN
Normal conversation.
Flirting.
Explicit sexual chat without payment.
Neutral contact exchange.
Consensual adult sexual discussion.

--------------------------------
DECISION FLOW:
1. Check explicit extreme danger → 6
2. Check clear payment/sales intent → 5
3. Check clear competitor bypass intent → 4
4. If ambiguous high-risk intent → 3
5. Check insults → 2
6. Otherwise → 1
`

const defaultSystemPromptSingleOutput = `Return compact JSON with minimal tokens:
{"a":status_code,"f":message_id,"c":confidence,"d":["token_or_phrase"]}`

const defaultSystemPromptBatchOutput = `Return compact JSON array with minimal tokens:
[{"a":status_code,"f":message_id,"c":confidence,"d":["token_or_phrase"]}]`

// DeepSeekAdapter is an HTTP AI adapter compatible with OpenAI-style chat completions.
type DeepSeekAdapter struct {
	baseURL      string
	model        string
	client       *resty.Client
	prompt       string
	customPrompt bool
	endpoint     string
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
	prompt := defaultSystemPromptBase + "\n" + defaultSystemPromptSingleOutput
	customPrompt := false
	if strings.TrimSpace(opt.SystemPrompt) != "" {
		prompt = opt.SystemPrompt
		customPrompt = true
	} else if strings.TrimSpace(opt.SystemHint) != "" {
		prompt = opt.SystemHint
		customPrompt = true
	}
	return &DeepSeekAdapter{
		baseURL:      strings.TrimRight(opt.BaseURL, "/"),
		model:        opt.Model,
		endpoint:     buildChatCompletionsURL(strings.TrimRight(opt.BaseURL, "/")),
		customPrompt: customPrompt,
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
			{Role: "system", Content: d.systemPromptFor(len(messages) > 1)},
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

func (d *DeepSeekAdapter) systemPromptFor(batch bool) string {
	if d.customPrompt {
		return d.prompt
	}
	if batch {
		return defaultSystemPromptBase + "\n" + defaultSystemPromptBatchOutput
	}
	return defaultSystemPromptBase + "\n" + defaultSystemPromptSingleOutput
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
				arr[i].StatusCode = models.StatusHumanReview
			}
		}
		return arr, nil
	}

	var one models.AIResult
	if err := json.Unmarshal([]byte(content), &one); err != nil {
		return nil, err
	}
	if !one.StatusCode.Valid() {
		one.StatusCode = models.StatusHumanReview
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
