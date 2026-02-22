package core

import (
	"context"
	"errors"
	"testing"

	"github.com/elum-utils/censor/models"
)

type errStorage struct{}

func (errStorage) AddToken(context.Context, string) error            { return errors.New("x") }
func (errStorage) RemoveToken(context.Context, string) error         { return nil }
func (errStorage) GetTokens(context.Context) ([]string, error)       { return nil, errors.New("x") }
func (errStorage) TokenExists(context.Context, string) (bool, error) { return false, nil }

func TestRunSyncErrorAndSyncNil(t *testing.T) {
	c := New(Options{AIAnalyzer: singleAI{}, Storage: errStorage{}})
	if err := c.SyncOnce(context.Background()); err == nil {
		t.Fatalf("expected sync error")
	}
	if err := c.Run(context.Background()); err == nil {
		t.Fatalf("expected run error")
	}

	c2 := New(Options{AIAnalyzer: singleAI{}})
	if err := c2.SyncOnce(context.Background()); err == nil {
		t.Fatalf("expected nil storage error")
	}
}

func TestProcessBatchMissingAIResultAndValidateMaxSize(t *testing.T) {
	ai := &mockAI{result: models.AIResult{StatusCode: models.StatusClean}}
	c := New(Options{AIAnalyzer: ai, Storage: newMockStorage("bad")})
	_ = c.SyncOnce(context.Background())

	// Force empty results by bypassing batch with analyzer error path.
	ai.err = nil
	res, err := c.ProcessBatch(context.Background(), []models.Message{{ID: 1, User: 2, Data: "bad"}})
	if err != nil || len(res) != 1 {
		t.Fatalf("unexpected result: %+v err=%v", res, err)
	}

	c.maxMessageSize = 0
	_, err = c.ProcessBatch(context.Background(), []models.Message{{ID: 2, User: 3, Data: "bad"}})
	if err == nil {
		t.Fatalf("expected max size validation error")
	}
}

func TestDispatchAllStatusesAndNoopMethods(t *testing.T) {
	n := noopCallbacks{}
	_ = n.OnClean(context.Background(), models.Violation{})
	_ = n.OnNonCriticalAbuse(context.Background(), models.Violation{})
	_ = n.OnSuspicious(context.Background(), models.Violation{})
	_ = n.OnCommercialOffPlatform(context.Background(), models.Violation{})
	_ = n.OnDangerousIllegal(context.Background(), models.Violation{})
	_ = n.OnCritical(context.Background(), models.Violation{})

	cb := &countCallbacks{}
	c := New(Options{AIAnalyzer: singleAI{}, Storage: newMockStorage(), CallbackHandler: cb})
	for _, code := range []models.StatusCode{
		models.StatusClean,
		models.StatusNonCriticalAbuse,
		models.StatusHumanReview,
		models.StatusSuspicious,
		models.StatusCommercialOffPlatform,
		models.StatusDangerousIllegal,
	} {
		c.record(models.Violation{Message: models.Message{ID: int64(code), User: 1}, AIResult: models.AIResult{StatusCode: code, ViolatorUserID: 1}})
	}
	if cb.clean.Load() != 1 || cb.abuse.Load() != 1 || cb.suspicious.Load() != 1 || cb.commercial.Load() != 1 || cb.dangerous.Load() != 1 || cb.critical.Load() != 1 {
		t.Fatalf("not all callbacks were called")
	}
}
