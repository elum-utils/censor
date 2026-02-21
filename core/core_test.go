package core

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/elum-utils/censor/interfaces"
	"github.com/elum-utils/censor/models"
)

type mockAI struct {
	batchCalled atomic.Bool
	result      models.AIResult
	err         error
}

func (m *mockAI) Name() string { return "mock" }
func (m *mockAI) Analyze(_ context.Context, msg models.Message) (models.AIResult, error) {
	if m.err != nil {
		return models.AIResult{}, m.err
	}
	res := m.result
	if res.MessageID == 0 {
		res.MessageID = msg.ID
	}
	if res.ViolatorUserID == 0 {
		res.ViolatorUserID = msg.User
	}
	return res, nil
}
func (m *mockAI) AnalyzeBatch(_ context.Context, msgs []models.Message) ([]models.AIResult, error) {
	m.batchCalled.Store(true)
	if m.err != nil {
		return nil, m.err
	}
	out := make([]models.AIResult, 0, len(msgs))
	for _, msg := range msgs {
		res := m.result
		res.MessageID = msg.ID
		res.ViolatorUserID = msg.User
		out = append(out, res)
	}
	return out, nil
}

var _ interfaces.BatchAIAnalyzer = (*mockAI)(nil)

type mockStorage struct {
	mu     sync.RWMutex
	tokens map[string]struct{}
}

func newMockStorage(tokens ...string) *mockStorage {
	m := &mockStorage{tokens: make(map[string]struct{}, len(tokens))}
	for _, t := range tokens {
		m.tokens[t] = struct{}{}
	}
	return m
}

func (m *mockStorage) AddToken(_ context.Context, token string) error {
	m.mu.Lock()
	m.tokens[token] = struct{}{}
	m.mu.Unlock()
	return nil
}
func (m *mockStorage) RemoveToken(_ context.Context, token string) error {
	m.mu.Lock()
	delete(m.tokens, token)
	m.mu.Unlock()
	return nil
}
func (m *mockStorage) GetTokens(context.Context) ([]string, error) {
	m.mu.RLock()
	out := make([]string, 0, len(m.tokens))
	for t := range m.tokens {
		out = append(out, t)
	}
	m.mu.RUnlock()
	return out, nil
}
func (m *mockStorage) TokenExists(_ context.Context, token string) (bool, error) {
	m.mu.RLock()
	_, ok := m.tokens[token]
	m.mu.RUnlock()
	return ok, nil
}

func (m *mockStorage) hasToken(token string) bool {
	m.mu.RLock()
	_, ok := m.tokens[token]
	m.mu.RUnlock()
	return ok
}

func TestProcessBatchNoTriggerSkipsAI(t *testing.T) {
	ai := &mockAI{result: models.AIResult{StatusCode: models.StatusCritical}}
	c := New(Options{AIAnalyzer: ai, Storage: newMockStorage("bad")})
	if err := c.SyncOnce(context.Background()); err != nil {
		t.Fatal(err)
	}

	res, err := c.ProcessBatch(context.Background(), []models.Message{{ID: 1, User: 2, Data: "hello"}})
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 1 || res[0].AIResult.StatusCode != models.StatusClean {
		t.Fatalf("unexpected result: %+v", res)
	}
	if ai.batchCalled.Load() {
		t.Fatalf("AI must not be called without triggers")
	}
}

func TestProcessBatchTriggerCallsAIAndLearns(t *testing.T) {
	ai := &mockAI{result: models.AIResult{StatusCode: models.StatusCommercialOffPlatform, Confidence: 0.9, TriggerTokens: []string{"new token"}}}
	st := newMockStorage("bad")
	c := New(Options{AIAnalyzer: ai, Storage: st, ConfidenceThreshold: 0.7, AutoLearn: true})
	if err := c.SyncOnce(context.Background()); err != nil {
		t.Fatal(err)
	}

	res, err := c.ProcessBatch(context.Background(), []models.Message{{ID: 1, User: 2, Data: "BAD words"}})
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 1 || res[0].AIResult.StatusCode != models.StatusCommercialOffPlatform {
		t.Fatalf("unexpected result: %+v", res)
	}

	time.Sleep(20 * time.Millisecond)
	if !st.hasToken("new token") {
		t.Fatalf("expected learned token persisted")
	}
}

func TestLowConfidenceNoLearn(t *testing.T) {
	ai := &mockAI{result: models.AIResult{StatusCode: models.StatusSuspicious, Confidence: 0.2, TriggerTokens: []string{"x"}}}
	st := newMockStorage("bad")
	c := New(Options{AIAnalyzer: ai, Storage: st, ConfidenceThreshold: 0.8, AutoLearn: true})
	_ = c.SyncOnce(context.Background())
	_, _ = c.ProcessBatch(context.Background(), []models.Message{{ID: 1, User: 2, Data: "bad"}})
	time.Sleep(20 * time.Millisecond)
	if st.hasToken("x") {
		t.Fatalf("token should not be learned")
	}
}

func TestAnalyzeError(t *testing.T) {
	ai := &mockAI{err: errors.New("boom")}
	c := New(Options{AIAnalyzer: ai, Storage: newMockStorage("bad")})
	_ = c.SyncOnce(context.Background())
	_, err := c.ProcessBatch(context.Background(), []models.Message{{ID: 1, User: 2, Data: "bad"}})
	if err == nil {
		t.Fatalf("expected error")
	}
}

func TestMaxSizeTrim(t *testing.T) {
	ai := &mockAI{result: models.AIResult{StatusCode: models.StatusSuspicious}}
	c := New(Options{AIAnalyzer: ai, Storage: newMockStorage("ab"), MaxMessageSize: 2})
	_ = c.SyncOnce(context.Background())
	res, err := c.ProcessBatch(context.Background(), []models.Message{{ID: 1, User: 2, Data: "ABCD"}})
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 1 || !res[0].Triggered {
		t.Fatalf("expected trigger after trim")
	}
}

func TestOnEvents(t *testing.T) {
	ai := &mockAI{result: models.AIResult{StatusCode: models.StatusCritical, Confidence: 1}}
	c := New(Options{AIAnalyzer: ai, Storage: newMockStorage("bad")})
	_ = c.SyncOnce(context.Background())

	var called atomic.Int64
	_ = c.On(EventCriticalEscalate, func(context.Context, ViolationEvent) error {
		called.Add(1)
		return nil
	})
	_, _ = c.ProcessBatch(context.Background(), []models.Message{{ID: 1, User: 2, Data: "bad"}})
	if called.Load() != 1 {
		t.Fatalf("expected event called once, got %d", called.Load())
	}
}

func TestConcurrentProcess(t *testing.T) {
	ai := &mockAI{result: models.AIResult{StatusCode: models.StatusSuspicious, Confidence: 1, TriggerTokens: []string{"x"}}}
	c := New(Options{AIAnalyzer: ai, Storage: newMockStorage("bad")})
	_ = c.SyncOnce(context.Background())

	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func(id int64) {
			defer wg.Done()
			_, _ = c.ProcessBatch(context.Background(), []models.Message{{ID: id, User: id, Data: "bad"}})
		}(int64(i + 1))
	}
	wg.Wait()
}

func TestProcessBatchWithOptionsSkipTriggerFilter(t *testing.T) {
	ai := &mockAI{result: models.AIResult{StatusCode: models.StatusSuspicious, Confidence: 0.8}}
	c := New(Options{AIAnalyzer: ai, Storage: newMockStorage("only-this-token")})
	if err := c.SyncOnce(context.Background()); err != nil {
		t.Fatal(err)
	}

	res, err := c.ProcessBatchWithOptions(
		context.Background(),
		[]models.Message{{ID: 1, User: 2, Data: "message without any token"}},
		ProcessOptions{SkipTriggerFilter: true},
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 1 {
		t.Fatalf("unexpected result len: %d", len(res))
	}
	if !ai.batchCalled.Load() {
		t.Fatalf("AI must be called when SkipTriggerFilter=true")
	}
	if res[0].Triggered {
		t.Fatalf("Triggered should be false when trigger pre-filter is bypassed")
	}
	if res[0].AIResult.StatusCode != models.StatusSuspicious {
		t.Fatalf("unexpected status: %d", res[0].AIResult.StatusCode)
	}
}
