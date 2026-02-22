package core

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/elum-utils/censor/engine"
	"github.com/elum-utils/censor/interfaces"
	"github.com/elum-utils/censor/models"
)

const (
	defaultConfidenceThreshold = 0.7
	defaultSyncInterval        = 5 * time.Minute
	defaultMaxMessageSize      = 4 * 1024
	defaultMaxLearnTokenLength = 255
	defaultCacheTTL            = 1 * time.Hour
	defaultCacheMaxBytes       = 32 * MB
)

// EventName is a callback bus event.
type EventName string

const (
	EventAllowClean       EventName = "allow_clean"
	EventMarkAbuse        EventName = "mark_abuse"
	EventHumanReview      EventName = "human_review"
	EventAutoRestrict     EventName = "auto_restrict"
	EventAutoBanEscalate  EventName = "auto_ban_escalate"
	EventCriticalEscalate EventName = "critical_escalate"
)

// ViolationEvent is callback payload.
type ViolationEvent struct {
	DialogID        string
	MessageID       int64
	ViolatorUserID  int64
	Reason          string
	Confidence      float64
	TriggerTokens   []string
	StatusCode      models.StatusCode
	TriggeredByRule bool
}

// EventHandler handles one moderation event.
type EventHandler func(ctx context.Context, event ViolationEvent) error

// ProcessOptions controls behavior of message checks.
type ProcessOptions struct {
	// SkipTriggerFilter forces AI analysis without in-memory trigger pre-filter.
	SkipTriggerFilter bool
}

// Options configure core filter.
type Options struct {
	AIAnalyzer      interfaces.AIAnalyzer
	Storage         interfaces.Storage
	CallbackHandler interfaces.CallbackHandler
	Processed       interfaces.ProcessedHandler
	Logger          interfaces.Logger

	ConfidenceThreshold float64
	SyncInterval        time.Duration
	MaxMessageSize      int
	MaxLearnTokenLength int
	CacheTTL            time.Duration
	CacheMaxBytes       int
	AutoLearn           bool
	DisableAutoLearn    bool
}

// Core is a two-level content filter.
type Core struct {
	ai      interfaces.AIAnalyzer
	storage interfaces.Storage
	cb      interfaces.CallbackHandler
	allCb   interfaces.ProcessedHandler
	logger  interfaces.Logger
	engine  *engine.Engine

	confidenceThreshold float64
	syncInterval        time.Duration
	maxMessageSize      int
	maxLearnTokenLength int
	negativeCacheTTL    time.Duration
	autoLearn           bool
	negativeCache       *negativeResultCache

	eventsMu sync.RWMutex
	events   map[EventName][]EventHandler

	processed [7]atomic.Int64
}

// New creates filter instance. Configuration errors are returned on Run/Process methods.
func New(opt Options) *Core {
	c := &Core{
		cb:                  noopCallbacks{},
		engine:              engine.New(),
		events:              make(map[EventName][]EventHandler, 6),
		confidenceThreshold: defaultConfidenceThreshold,
		syncInterval:        defaultSyncInterval,
		maxMessageSize:      defaultMaxMessageSize,
		maxLearnTokenLength: defaultMaxLearnTokenLength,
		negativeCacheTTL:    defaultCacheTTL,
		autoLearn:           true,
	}

	if opt.ConfidenceThreshold > 0 {
		c.confidenceThreshold = opt.ConfidenceThreshold
	}
	if opt.SyncInterval > 0 {
		c.syncInterval = opt.SyncInterval
	}
	if opt.MaxMessageSize > 0 {
		c.maxMessageSize = opt.MaxMessageSize
	}
	if opt.MaxLearnTokenLength > 0 {
		c.maxLearnTokenLength = opt.MaxLearnTokenLength
	}
	if opt.CacheTTL > 0 {
		c.negativeCacheTTL = opt.CacheTTL
	}
	cacheMaxBytes := defaultCacheMaxBytes
	if opt.CacheMaxBytes > 0 {
		cacheMaxBytes = opt.CacheMaxBytes
	}
	if opt.AutoLearn {
		c.autoLearn = true
	}
	if opt.DisableAutoLearn {
		c.autoLearn = false
	}
	if opt.Logger != nil {
		c.logger = opt.Logger
	}
	if opt.CallbackHandler != nil {
		c.cb = opt.CallbackHandler
	}
	if opt.Processed != nil {
		c.allCb = opt.Processed
	}

	c.ai = opt.AIAnalyzer
	c.storage = opt.Storage
	c.negativeCache = newNegativeResultCache(int64(cacheMaxBytes))
	c.startNegativeCacheJanitor()

	return c
}

// On registers event handlers.
func (c *Core) On(event EventName, handler EventHandler) error {
	if handler == nil {
		return errors.New("core: handler is nil")
	}
	c.eventsMu.Lock()
	c.events[event] = append(c.events[event], handler)
	c.eventsMu.Unlock()
	return nil
}

// OnAllowClean registers handler for status code 1 (clean).
func (c *Core) OnAllowClean(handler EventHandler) error {
	return c.On(EventAllowClean, handler)
}

// OnMarkAbuse registers handler for status code 2 (non-critical abuse).
func (c *Core) OnMarkAbuse(handler EventHandler) error {
	return c.On(EventMarkAbuse, handler)
}

// OnHumanReview registers handler for status code 3 (suspicious).
func (c *Core) OnHumanReview(handler EventHandler) error {
	return c.On(EventHumanReview, handler)
}

// OnAutoRestrict registers handler for status code 4 (commercial/off-platform).
func (c *Core) OnAutoRestrict(handler EventHandler) error {
	return c.On(EventAutoRestrict, handler)
}

// OnAutoBanEscalate registers handler for status code 5 (dangerous/illegal).
func (c *Core) OnAutoBanEscalate(handler EventHandler) error {
	return c.On(EventAutoBanEscalate, handler)
}

// OnCriticalEscalate registers handler for status code 6 (critical).
func (c *Core) OnCriticalEscalate(handler EventHandler) error {
	return c.On(EventCriticalEscalate, handler)
}

// Run loads initial tokens and starts periodic sync until context cancellation.
func (c *Core) Run(ctx context.Context) error {
	if err := c.validate(); err != nil {
		return err
	}
	if err := c.SyncOnce(ctx); err != nil {
		return err
	}

	ticker := time.NewTicker(c.syncInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			if err := c.SyncOnce(ctx); err != nil {
				c.logWarn("sync failed", map[string]any{"error": err.Error()})
			}
		}
	}
}

// SyncOnce reloads token set from storage.
func (c *Core) SyncOnce(ctx context.Context) error {
	if c.storage == nil {
		return errors.New("core: storage is nil")
	}
	tokens, err := c.storage.GetTokens(ctx)
	if err != nil {
		return err
	}
	c.engine.ReplaceAll(tokens)
	return nil
}

// ProcessMessage processes one message.
func (c *Core) ProcessMessage(ctx context.Context, message models.Message) (models.Violation, error) {
	return c.ProcessMessageWithOptions(ctx, message, ProcessOptions{})
}

// ProcessMessageWithOptions processes one message with custom process behavior.
func (c *Core) ProcessMessageWithOptions(ctx context.Context, message models.Message, opt ProcessOptions) (models.Violation, error) {
	res, err := c.ProcessBatchWithOptions(ctx, []models.Message{message}, opt)
	if err != nil {
		return models.Violation{}, err
	}
	if len(res) == 0 {
		return models.Violation{}, errors.New("core: empty result")
	}
	return res[0], nil
}

// ProcessBatch processes multiple messages with trigger pre-filter and AI stage.
func (c *Core) ProcessBatch(ctx context.Context, messages []models.Message) ([]models.Violation, error) {
	return c.ProcessBatchWithOptions(ctx, messages, ProcessOptions{})
}

// ProcessBatchWithOptions processes multiple messages with custom process behavior.
func (c *Core) ProcessBatchWithOptions(ctx context.Context, messages []models.Message, opt ProcessOptions) ([]models.Violation, error) {
	if err := c.validate(); err != nil {
		return nil, err
	}
	if len(messages) == 0 {
		return nil, nil
	}

	type pendingAnalyze struct {
		index    int
		message  models.Message
		triggers []string
	}

	out := make([]models.Violation, len(messages))
	filled := make([]bool, len(messages))
	toAnalyze := make([]pendingAnalyze, 0, len(messages))

	for i, msg := range messages {
		prepared := msg
		if len(prepared.Data) > c.maxMessageSize {
			prepared.Data = prepared.Data[:c.maxMessageSize]
		}
		cacheKey := prepared.Data
		if opt.SkipTriggerFilter {
			if cached, ok := c.getCachedNegative(cacheKey, prepared); ok {
				v := models.Violation{Message: prepared, Triggered: false, AIResult: cached}
				c.record(v)
				out[i] = v
				filled[i] = true
				continue
			}
			toAnalyze = append(toAnalyze, pendingAnalyze{index: i, message: prepared, triggers: nil})
			continue
		}
		triggers := c.engine.FindTriggers(prepared.Data)
		if len(triggers) == 0 {
			v := models.Violation{Message: prepared, Triggered: false, AIResult: models.AIResult{
				StatusCode:     models.StatusClean,
				Reason:         "no trigger",
				Confidence:     1,
				ViolatorUserID: prepared.User,
				MessageID:      prepared.ID,
			}}
			c.record(v)
			out[i] = v
			filled[i] = true
			continue
		}
		if cached, ok := c.getCachedNegative(cacheKey, prepared); ok {
			if len(cached.TriggerTokens) == 0 {
				cached.TriggerTokens = triggers
			}
			v := models.Violation{Message: prepared, Triggered: true, AIResult: cached}
			c.record(v)
			out[i] = v
			filled[i] = true
			continue
		}
		toAnalyze = append(toAnalyze, pendingAnalyze{index: i, message: prepared, triggers: triggers})
	}

	if len(toAnalyze) == 0 {
		return out, nil
	}

	aiMessages := make([]models.Message, 0, len(toAnalyze))
	for _, p := range toAnalyze {
		aiMessages = append(aiMessages, p.message)
	}
	results, err := c.analyze(ctx, aiMessages)
	if err != nil {
		return nil, err
	}

	byID := make(map[int64]models.AIResult, len(results))
	for _, r := range results {
		byID[r.MessageID] = r
	}
	for _, p := range toAnalyze {
		msg := p.message
		r, ok := byID[msg.ID]
		if !ok {
			r = models.AIResult{
				StatusCode:     models.StatusSuspicious,
				Reason:         "missing AI result",
				Confidence:     0,
				TriggerTokens:  p.triggers,
				ViolatorUserID: msg.User,
				MessageID:      msg.ID,
			}
		}
		if r.ViolatorUserID == 0 {
			r.ViolatorUserID = msg.User
		}
		if r.MessageID == 0 {
			r.MessageID = msg.ID
		}
		if len(r.TriggerTokens) == 0 {
			r.TriggerTokens = p.triggers
		}
		v := models.Violation{Message: msg, Triggered: len(p.triggers) > 0, AIResult: r}
		c.setCachedNegative(msg.Data, r)
		c.learn(r)
		c.record(v)
		out[p.index] = v
		filled[p.index] = true
	}

	for i := range out {
		if !filled[i] {
			return nil, fmt.Errorf("core: internal batch result mismatch at index %d", i)
		}
	}
	return out, nil
}

func (c *Core) analyze(ctx context.Context, messages []models.Message) ([]models.AIResult, error) {
	if batch, ok := c.ai.(interfaces.BatchAIAnalyzer); ok {
		return batch.AnalyzeBatch(ctx, messages)
	}
	out := make([]models.AIResult, 0, len(messages))
	for _, message := range messages {
		res, err := c.ai.Analyze(ctx, message)
		if err != nil {
			return nil, err
		}
		if res.MessageID == 0 {
			res.MessageID = message.ID
		}
		if res.ViolatorUserID == 0 {
			res.ViolatorUserID = message.User
		}
		out = append(out, res)
	}
	return out, nil
}

func (c *Core) learn(result models.AIResult) {
	if !c.autoLearn || c.storage == nil {
		return
	}
	if result.Confidence < c.confidenceThreshold {
		return
	}
	for _, token := range result.TriggerTokens {
		normalized := strings.ToLower(strings.TrimSpace(token))
		if normalized == "" {
			continue
		}
		if len(normalized) > c.maxLearnTokenLength {
			c.logWarn("token exceeds max learn length", map[string]any{
				"token":      normalized,
				"length":     len(normalized),
				"max_length": c.maxLearnTokenLength,
			})
			continue
		}
		if !c.engine.AddToken(normalized) {
			continue
		}
		go func(tok string) {
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()
			if err := c.storage.AddToken(ctx, tok); err != nil {
				c.logWarn("token persist failed", map[string]any{"error": err.Error(), "token": tok})
			}
		}(normalized)
	}
}

// Metrics returns count of processed messages by status code 1..6.
func (c *Core) Metrics() map[models.StatusCode]int64 {
	out := make(map[models.StatusCode]int64, 6)
	for i := 1; i <= 6; i++ {
		out[models.StatusCode(i)] = c.processed[i].Load()
	}
	return out
}

// TokenCount returns number of in-memory tokens.
func (c *Core) TokenCount() int {
	return c.engine.Count()
}

func (c *Core) record(v models.Violation) {
	code := v.AIResult.StatusCode
	if !code.Valid() {
		code = models.StatusSuspicious
	}
	c.processed[code].Add(1)
	e := ViolationEvent{
		DialogID:        v.Message.DialogID,
		MessageID:       v.Message.ID,
		ViolatorUserID:  v.AIResult.ViolatorUserID,
		Reason:          v.AIResult.Reason,
		Confidence:      v.AIResult.Confidence,
		TriggerTokens:   v.AIResult.TriggerTokens,
		StatusCode:      code,
		TriggeredByRule: v.Triggered,
	}
	c.dispatchByStatus(context.Background(), e)
	c.dispatchEvent(context.Background(), e)
}

func (c *Core) dispatchByStatus(ctx context.Context, e ViolationEvent) {
	var err error
	switch e.StatusCode {
	case models.StatusClean:
		err = c.cb.OnClean(ctx, toViolation(e))
	case models.StatusNonCriticalAbuse:
		err = c.cb.OnNonCriticalAbuse(ctx, toViolation(e))
	case models.StatusSuspicious:
		err = c.cb.OnSuspicious(ctx, toViolation(e))
	case models.StatusCommercialOffPlatform:
		err = c.cb.OnCommercialOffPlatform(ctx, toViolation(e))
	case models.StatusDangerousIllegal:
		err = c.cb.OnDangerousIllegal(ctx, toViolation(e))
	case models.StatusCritical:
		err = c.cb.OnCritical(ctx, toViolation(e))
	}
	if err != nil {
		c.logWarn("callback failed", map[string]any{"error": err.Error(), "status": e.StatusCode})
	}
	if c.allCb != nil {
		if err := c.allCb.OnProcessed(ctx, toViolation(e)); err != nil {
			c.logWarn("processed callback failed", map[string]any{"error": err.Error()})
		}
	}
}

func (c *Core) dispatchEvent(ctx context.Context, e ViolationEvent) {
	event := eventNameFromCode(e.StatusCode)
	c.eventsMu.RLock()
	handlers := append([]EventHandler(nil), c.events[event]...)
	c.eventsMu.RUnlock()
	for _, h := range handlers {
		if err := h(ctx, e); err != nil {
			c.logWarn("event handler failed", map[string]any{"error": err.Error(), "event": event})
		}
	}
}

func eventNameFromCode(code models.StatusCode) EventName {
	switch code {
	case models.StatusClean:
		return EventAllowClean
	case models.StatusNonCriticalAbuse:
		return EventMarkAbuse
	case models.StatusSuspicious:
		return EventHumanReview
	case models.StatusCommercialOffPlatform:
		return EventAutoRestrict
	case models.StatusDangerousIllegal:
		return EventAutoBanEscalate
	case models.StatusCritical:
		return EventCriticalEscalate
	default:
		return EventHumanReview
	}
}

func (c *Core) validate() error {
	if c.ai == nil {
		return errors.New("core: AI analyzer is nil")
	}
	if c.storage == nil {
		return errors.New("core: storage is nil")
	}
	if c.maxMessageSize <= 0 {
		return fmt.Errorf("core: invalid max message size: %d", c.maxMessageSize)
	}
	return nil
}

func toViolation(e ViolationEvent) models.Violation {
	return models.Violation{
		Message: models.Message{ID: e.MessageID, DialogID: e.DialogID, User: e.ViolatorUserID},
		AIResult: models.AIResult{
			StatusCode:     e.StatusCode,
			Reason:         e.Reason,
			Confidence:     e.Confidence,
			TriggerTokens:  e.TriggerTokens,
			ViolatorUserID: e.ViolatorUserID,
			MessageID:      e.MessageID,
		},
		Triggered: e.TriggeredByRule,
	}
}

func (c *Core) logWarn(msg string, fields map[string]any) {
	if c.logger != nil {
		c.logger.Warn(msg, fields)
	}
}

func (c *Core) getCachedNegative(key string, message models.Message) (models.AIResult, bool) {
	if c.negativeCache == nil {
		return models.AIResult{}, false
	}
	res, ok := c.negativeCache.Get(key, time.Now())
	if !ok {
		return models.AIResult{}, false
	}
	res.MessageID = message.ID
	res.ViolatorUserID = message.User
	return res, true
}

func (c *Core) setCachedNegative(key string, result models.AIResult) {
	if c.negativeCache == nil || key == "" {
		return
	}
	if !result.StatusCode.Valid() {
		return
	}
	c.negativeCache.Set(key, result, c.negativeCacheTTL, time.Now())
}

func (c *Core) startNegativeCacheJanitor() {
	if c.negativeCache == nil {
		return
	}
	interval := time.Minute
	if c.negativeCacheTTL > 0 && c.negativeCacheTTL < interval {
		interval = c.negativeCacheTTL
	}
	go func() {
		defer func() {
			if r := recover(); r != nil {
				c.logWarn("negative cache janitor panic", map[string]any{"panic": fmt.Sprint(r)})
			}
		}()

		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for range ticker.C {
			func() {
				defer func() {
					if r := recover(); r != nil {
						c.logWarn("negative cache sweep panic", map[string]any{"panic": fmt.Sprint(r)})
					}
				}()
				c.negativeCache.RemoveExpired(time.Now())
			}()
		}
	}()
}

type noopCallbacks struct{}

func (noopCallbacks) OnClean(context.Context, models.Violation) error                 { return nil }
func (noopCallbacks) OnNonCriticalAbuse(context.Context, models.Violation) error      { return nil }
func (noopCallbacks) OnSuspicious(context.Context, models.Violation) error            { return nil }
func (noopCallbacks) OnCommercialOffPlatform(context.Context, models.Violation) error { return nil }
func (noopCallbacks) OnDangerousIllegal(context.Context, models.Violation) error      { return nil }
func (noopCallbacks) OnCritical(context.Context, models.Violation) error              { return nil }
