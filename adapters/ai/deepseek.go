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

--------------------------------
IMPORTANT DISTINCTION (SELLER vs BUYER):

- Distinguish between seller and buyer roles.
- Only the seller (who offers or initiates payment/content exchange) can trigger level 5.
- Buyer-side messages alone must NOT trigger level 5.

Level 5 applies ONLY to the party initiating or offering a commercial transaction.

Do NOT use level 5 for buyer behavior:
- Asking about price
- Requesting preview before paying
- Showing interest in buying
- Evaluating what is being sold

Examples that are NOT level 5:
- "сколько стоит?"
- "покажи перед оплатой"
- "за что платить?"
- "что входит?"
- "чтоб видел за что плачу"

These should be level 1 or level 3 depending on clarity.

--------------------------------
Codes:
1 clean
2 non-critical abuse
3 human review required
4 suspicious competitor bypass
5 commercial / selling / paid off-platform
6 dangerous / illegal (extreme only)

--------------------------------
GENERAL PRINCIPLES:

- Sexual and intimate conversations between consenting adults are allowed.
- Explicit sexual content alone is NOT a violation.
- Neutral contact exchange (Telegram, Instagram, etc.) is allowed.
- Detect intent, not keywords alone.
- Context matters, but do NOT over-infer.
- For levels 1-3 omit triggers.
- For levels 4-6 include short trigger tokens (max 255 chars each).

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

--------------------------------
LEVEL 5 — COMMERCIAL / PAID (SELLER ONLY)

Use ONLY if:
- User INITIATES selling or payment exchange
- Offers content/service for money
- Mentions price / payment / transaction
- Redirects to another platform for payment

Examples:
- "продаю фото"
- "скинь деньги — покажу"
- "прайс в тг"
- "пиши в тг для покупки"

If payment intent is unclear → DO NOT use 5.

--------------------------------
LEVEL 4 — COMPETITOR BYPASS

Use ONLY if:
- User says platform is worse
- Encourages leaving platform explicitly
- Mentions bypassing moderation

Do NOT use for:
- "давай в тг"
- username sharing

--------------------------------
LEVEL 3 — HUMAN REVIEW

Use when:
- Possible payment but unclear
- Possible selling but unclear
- Ambiguous intent

--------------------------------
LEVEL 2 — NON-CRITICAL ABUSE

Insults, rude language, harassment without real threat.

--------------------------------
LEVEL 1 — CLEAN

- Normal conversation
- Flirting
- Explicit sexual chat (no payment)
- Buyer behavior
- Neutral contact exchange

--------------------------------
FEW-SHOT EXAMPLES:

Message:
"продаю фото и видео, интересует?"
Output:
{"a":5,"c":0.95,"d":["продаю","видео"]}

Message:
"скинешь деньги — покажу"
Output:
{"a":5,"c":0.97,"d":["деньги","покажу"]}

Message:
"д22 скинешь на вкусняшки?, а я тебе себя покажу?)"
Output:
{"a":5,"c":0.94,"d":["скинешь","покажу"]}

Message:
"чтоб видел за что плачу"
Output:
{"a":1,"c":0.90,"d":[]}

Message:
"покажи перед оплатой"
Output:
{"a":1,"c":0.90,"d":[]}

Message:
"за что платить?"
Output:
{"a":1,"c":0.90,"d":[]}

Message:
"сколько стоит?"
Output:
{"a":1,"c":0.90,"d":[]}

Message:
"покажи фото"
Output:
{"a":1,"c":0.88,"d":[]}

Message:
"давай в тг"
Output:
{"a":1,"c":0.85,"d":[]}

Message:
"этот сайт говно, пиши в тг"
Output:
{"a":4,"c":0.92,"d":["говно","в тг"]}

--------------------------------
DECISION FLOW:

1. Explicit extreme danger → 6
2. Clear seller payment intent → 5
3. Clear competitor bypass → 4
4. Ambiguous high-risk → 3
5. Abuse → 2
6. Otherwise → 1
`

const defaultSystemPromptSingleOutput = `
Return compact JSON:
{"a":status_code,"f":message_id,"c":confidence,"d":["token"]}
`

const defaultSystemPromptBatchOutput = `
Return compact JSON array:
[{"a":status_code,"f":message_id,"c":confidence,"d":["token"]}]
`

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
