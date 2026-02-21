package core

import (
	"context"
	"testing"

	"github.com/elum-utils/censor/models"
)

func TestOnConvenienceMethods(t *testing.T) {
	c := New(Options{AIAnalyzer: singleAI{}, Storage: newMockStorage()})

	if err := c.OnAllowClean(func(context.Context, ViolationEvent) error { return nil }); err != nil {
		t.Fatal(err)
	}
	if err := c.OnMarkAbuse(func(context.Context, ViolationEvent) error { return nil }); err != nil {
		t.Fatal(err)
	}
	if err := c.OnHumanReview(func(context.Context, ViolationEvent) error { return nil }); err != nil {
		t.Fatal(err)
	}
	if err := c.OnAutoRestrict(func(context.Context, ViolationEvent) error { return nil }); err != nil {
		t.Fatal(err)
	}
	if err := c.OnAutoBanEscalate(func(context.Context, ViolationEvent) error { return nil }); err != nil {
		t.Fatal(err)
	}
	if err := c.OnCriticalEscalate(func(context.Context, ViolationEvent) error { return nil }); err != nil {
		t.Fatal(err)
	}

	for _, code := range []models.StatusCode{
		models.StatusClean,
		models.StatusNonCriticalAbuse,
		models.StatusSuspicious,
		models.StatusCommercialOffPlatform,
		models.StatusDangerousIllegal,
		models.StatusCritical,
	} {
		c.record(models.Violation{Message: models.Message{ID: int64(code), User: 1}, AIResult: models.AIResult{StatusCode: code, ViolatorUserID: 1}})
	}
}
