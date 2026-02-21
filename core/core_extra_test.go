package core

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/elum-utils/censor/models"
)

type singleAI struct {
	res models.AIResult
	err error
}

func (s singleAI) Name() string { return "single" }
func (s singleAI) Analyze(context.Context, models.Message) (models.AIResult, error) {
	if s.err != nil {
		return models.AIResult{}, s.err
	}
	return s.res, nil
}

type countCallbacks struct {
	clean, abuse, suspicious, commercial, dangerous, critical atomic.Int64
}

func (c *countCallbacks) OnClean(context.Context, models.Violation) error { c.clean.Add(1); return nil }
func (c *countCallbacks) OnNonCriticalAbuse(context.Context, models.Violation) error {
	c.abuse.Add(1)
	return nil
}
func (c *countCallbacks) OnSuspicious(context.Context, models.Violation) error {
	c.suspicious.Add(1)
	return nil
}
func (c *countCallbacks) OnCommercialOffPlatform(context.Context, models.Violation) error {
	c.commercial.Add(1)
	return nil
}
func (c *countCallbacks) OnDangerousIllegal(context.Context, models.Violation) error {
	c.dangerous.Add(1)
	return nil
}
func (c *countCallbacks) OnCritical(context.Context, models.Violation) error {
	c.critical.Add(1)
	return nil
}

type noopProcessed struct{ called atomic.Int64 }

func (n *noopProcessed) OnProcessed(context.Context, models.Violation) error {
	n.called.Add(1)
	return nil
}

type testLogger struct{ warned atomic.Int64 }

func (l *testLogger) Debug(string, map[string]any) {}
func (l *testLogger) Info(string, map[string]any)  {}
func (l *testLogger) Warn(string, map[string]any)  { l.warned.Add(1) }
func (l *testLogger) Error(string, map[string]any) {}

func TestValidateErrors(t *testing.T) {
	_, err := New(Options{}).ProcessBatch(context.Background(), []models.Message{{ID: 1, User: 1, Data: "x"}})
	if err == nil {
		t.Fatalf("expected validation error")
	}

	c := New(Options{AIAnalyzer: singleAI{}})
	_, err = c.ProcessBatch(context.Background(), []models.Message{{ID: 1, User: 1, Data: "x"}})
	if err == nil {
		t.Fatalf("expected storage validation error")
	}
}

func TestProcessMessageAndMetrics(t *testing.T) {
	st := newMockStorage("trigger")
	cbs := &countCallbacks{}
	processed := &noopProcessed{}
	c := New(Options{
		AIAnalyzer:      singleAI{res: models.AIResult{StatusCode: models.StatusDangerousIllegal, Confidence: 1}},
		Storage:         st,
		CallbackHandler: cbs,
		Processed:       processed,
	})
	if err := c.SyncOnce(context.Background()); err != nil {
		t.Fatal(err)
	}
	res, err := c.ProcessMessage(context.Background(), models.Message{ID: 7, User: 3, Data: "trigger"})
	if err != nil {
		t.Fatal(err)
	}
	if res.AIResult.StatusCode != models.StatusDangerousIllegal {
		t.Fatalf("unexpected status: %d", res.AIResult.StatusCode)
	}
	if c.TokenCount() != 1 {
		t.Fatalf("unexpected token count: %d", c.TokenCount())
	}
	m := c.Metrics()
	if m[models.StatusDangerousIllegal] != 1 {
		t.Fatalf("unexpected metrics: %+v", m)
	}
	if cbs.dangerous.Load() != 1 || processed.called.Load() != 1 {
		t.Fatalf("callbacks not called")
	}
}

func TestAnalyzeFallbackNonBatch(t *testing.T) {
	st := newMockStorage("x")
	c := New(Options{AIAnalyzer: singleAI{res: models.AIResult{StatusCode: models.StatusClean}}, Storage: st})
	_ = c.SyncOnce(context.Background())
	res, err := c.ProcessBatch(context.Background(), []models.Message{{ID: 1, User: 2, Data: "x"}})
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 1 {
		t.Fatalf("unexpected result len")
	}
}

func TestRunContextCancel(t *testing.T) {
	st := newMockStorage("a")
	c := New(Options{AIAnalyzer: singleAI{res: models.AIResult{StatusCode: models.StatusClean}}, Storage: st, SyncInterval: time.Millisecond})
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(5 * time.Millisecond)
		cancel()
	}()
	if err := c.Run(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("unexpected err: %v", err)
	}
}

func TestOnNilHandlerAndLogWarn(t *testing.T) {
	l := &testLogger{}
	c := New(Options{AIAnalyzer: singleAI{}, Storage: newMockStorage(), Logger: l})
	if err := c.On(EventAllowClean, nil); err == nil {
		t.Fatalf("expected error")
	}
	_ = c.On(EventAllowClean, func(context.Context, ViolationEvent) error { return errors.New("x") })
	c.record(models.Violation{Message: models.Message{ID: 1, User: 1}, AIResult: models.AIResult{StatusCode: models.StatusClean, ViolatorUserID: 1}})
	if l.warned.Load() == 0 {
		t.Fatalf("expected warning logs")
	}
}

func TestEventMappingDefault(t *testing.T) {
	if eventNameFromCode(models.StatusCode(42)) != EventHumanReview {
		t.Fatalf("unexpected mapping")
	}
}
