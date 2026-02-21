package storage

import (
	"context"
	"sync"
)

// MemoryAdapter is an in-memory storage implementation.
type MemoryAdapter struct {
	mu     sync.RWMutex
	tokens map[string]struct{}
}

// NewMemoryAdapter creates a memory storage adapter.
func NewMemoryAdapter() *MemoryAdapter {
	return &MemoryAdapter{tokens: make(map[string]struct{})}
}

func (m *MemoryAdapter) AddToken(_ context.Context, token string) error {
	m.mu.Lock()
	m.tokens[token] = struct{}{}
	m.mu.Unlock()
	return nil
}

func (m *MemoryAdapter) RemoveToken(_ context.Context, token string) error {
	m.mu.Lock()
	delete(m.tokens, token)
	m.mu.Unlock()
	return nil
}

func (m *MemoryAdapter) GetTokens(_ context.Context) ([]string, error) {
	m.mu.RLock()
	out := make([]string, 0, len(m.tokens))
	for token := range m.tokens {
		out = append(out, token)
	}
	m.mu.RUnlock()
	return out, nil
}

func (m *MemoryAdapter) TokenExists(_ context.Context, token string) (bool, error) {
	m.mu.RLock()
	_, ok := m.tokens[token]
	m.mu.RUnlock()
	return ok, nil
}
