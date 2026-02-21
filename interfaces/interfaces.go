package interfaces

import (
	"context"

	"github.com/elum-utils/censor/models"
)

// AIAnalyzer analyzes a single message.
type AIAnalyzer interface {
	Name() string
	Analyze(ctx context.Context, message models.Message) (models.AIResult, error)
}

// BatchAIAnalyzer extends AIAnalyzer with batch processing.
type BatchAIAnalyzer interface {
	AIAnalyzer
	AnalyzeBatch(ctx context.Context, messages []models.Message) ([]models.AIResult, error)
}

// Storage persists trigger tokens.
type Storage interface {
	AddToken(ctx context.Context, token string) error
	RemoveToken(ctx context.Context, token string) error
	GetTokens(ctx context.Context) ([]string, error)
	TokenExists(ctx context.Context, token string) (bool, error)
}

// CallbackHandler handles results by status code.
type CallbackHandler interface {
	OnClean(ctx context.Context, event models.Violation) error
	OnNonCriticalAbuse(ctx context.Context, event models.Violation) error
	OnSuspicious(ctx context.Context, event models.Violation) error
	OnCommercialOffPlatform(ctx context.Context, event models.Violation) error
	OnDangerousIllegal(ctx context.Context, event models.Violation) error
	OnCritical(ctx context.Context, event models.Violation) error
}

// ProcessedHandler handles every result with one method.
type ProcessedHandler interface {
	OnProcessed(ctx context.Context, event models.Violation) error
}

// Logger is an optional structured logger.
type Logger interface {
	Debug(msg string, fields map[string]any)
	Info(msg string, fields map[string]any)
	Warn(msg string, fields map[string]any)
	Error(msg string, fields map[string]any)
}
